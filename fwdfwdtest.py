import pickle
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

device = "cuda"

class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(input):
        return input.clamp(min=0)

    @staticmethod
    def backward(grad_output):
        return grad_output.clone()

class FwdFwdModel(torch.nn.Module):
    def __init__(self, layers):
        super(FwdFwdModel, self).__init__()

        self.act_fn = nn.ReLU()

        self.model = nn.ModuleList([])
        self.optimizers = []
        for i in range(len(layers) - 1):
            self.model.append(nn.Linear(layers[i], layers[i+1]))
            self.optimizers.append(torch.optim.Adam(self.model[len(self.model) - 1].parameters(), lr=0.01))

        self.ff_loss = nn.BCEWithLogitsLoss()

        self.linear_classifer = nn.Sequential(
            # TODO (should we omit the first Linear from classifier?)
            nn.Linear(sum(layers[1:]), 10, bias=False) # 10 = the number of classes
        )
        self.classification_loss = nn.CrossEntropyLoss()

        self._init_weights()

    def _init_weights(self):
        # TODO (understand this param init fn)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0]))
            # torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifer.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _calc_ff_loss(self, z, labels, k=1):
        sum_of_squares = torch.sum(z ** 2, dim=-1)
        logits = sum_of_squares - z.shape[-1]*k
        logits = torch.reshape(torch.sigmoid(logits), (len(z), 1))
        ff_loss = self.ff_loss(logits, labels)

        return ff_loss, sum_of_squares

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _linear_classifier_fwd(self, input, label):
        # append all weight tensors into single tensor
        neural_sample = input[0]
        for i in range(1, len(input)):
            # TODO (line below might break when batch size > 1)
            neural_sample = torch.cat((neural_sample, input[i]), -1)

        # forward pass through classifier
        output = self.linear_classifer(neural_sample.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0] # TODO (not entirely clear what this is for)
        
        # loss
        classification_loss = self.classification_loss(output, label*1.0)

        # return
        return classification_loss, output

    def forward(self, inputs, ff_labels, class_labels):
        # scalar_outputs = {
        #     "Loss": torch.zeros(1, device="cpu")
        # }


        z = inputs

        optim_idx = 0
        neural_sample = []
        for idx, layer in enumerate(self.model):
            z = z.detach()  # Detach to ensure no computation graph reuse
            z.requires_grad_()
            z = layer(z)
            z = self.act_fn(z) # forward through layer
            neural_sample.append(z)
            ff_loss, _ = self._calc_ff_loss(z, ff_labels, k=0.5) # calc layer wise loss
            self.optimizers[optim_idx].zero_grad()

            ff_loss.backward() # compute gradients for layer

            self.optimizers[optim_idx].step() # step forward
            optim_idx += 1

            z = self._layer_norm(z) # normalize for next layer


        # only do the fwd pass on linear classifier if the data is positive
        if ff_labels[0]:
            return self._linear_classifier_fwd(neural_sample, class_labels)
        return None, None

    def get_linear_classifier_param(self):
        return self.linear_classifer.parameters()

    def infer(self, inputs):
        z = inputs
        neural_sample = []
        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn(z)
                z = self._layer_norm(z)
                neural_sample.append(z)
            lr_input = neural_sample[0]
            for i in range(1, len(neural_sample)):
                lr_input = torch.cat((lr_input, neural_sample[i]), -1)
            output = self.linear_classifer(lr_input)
            output = output - torch.max(output, dim=-1, keepdim=True)[0]
            return output

    def get_goodness(self, inputs):
        with torch.no_grad():
            sum_of_squares = torch.zeros(len(inputs)).to(device)
            z = inputs
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn(z)
                z = self._layer_norm(z)
                sum_of_squares += torch.sum(z ** 2, dim=-1)
        return sum_of_squares.reshape((len(inputs), 1))

    def slower_inference(self, inputs):
        predictions = torch.full((len(inputs), 1), float('-inf')).to(device)
        for i in range(0, 10):
            labels = F.one_hot((torch.ones((len(inputs))) * i).to(torch.int64), num_classes=10)
            label_and_input, _ = preprocess_sample(inputs, labels)
            prediction = self.get_goodness(label_and_input)
            predictions = torch.cat((predictions, prediction), -1)
        return torch.argmax(predictions, -1)

def preprocess_sample(inputs, labels):
    inputs = inputs.to(device)
    labels = labels.to(device)
    if labels[0].shape != torch.Size([10]):
        one_hot_labels = F.one_hot(labels, num_classes=10)
    else:
        one_hot_labels = labels
    inputs = torch.reshape(inputs, (len(inputs), 784))
    # concatinate the label as a 1 hot encoding and flattened image
    return torch.cat((one_hot_labels, inputs), -1), one_hot_labels * 1.0

def create_negative_label(labels):
    labels = labels.detach().clone()
    labels = labels.to(device)
    for i in range(len(labels)):
        neg_label = random.randint(0, 9)
        while neg_label == labels[i].item():
            neg_label = random.randint(0, 9)
        labels[i] = neg_label
    return labels

def train(epochs, model, optimizer, train_loader): # this optimizer and loss is for the linear classifier

    for epoch in range(epochs):
        epoch_accuracy = 0
        sample_counter = 0 # for now ill include negative samples in this too
        for inputs, labels, in train_loader:

            # positive data forward ~~~~~~~~~~
            optimizer.zero_grad()

            # making the label 1 hot encoding
            label_and_input, one_hot_labels = preprocess_sample(inputs, labels)
            ff_labels = torch.ones(len(inputs), 1).to(device)
            loss, output = model(label_and_input, ff_labels, one_hot_labels) 
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # negative data forward ~~~~~~~~~~
            # TODO (Note: pretty sure using model output -> neg data pipeline is a good idea anymore)
            # optimizer.zero_grad()

            # update accuracy calcs after positive data step
            sample_counter += len(inputs)
            epoch_accuracy += absolute_loss(torch.argmax(one_hot_labels, -1), output)

            negative_labels = create_negative_label(labels) # shuffles around labels
            label_and_input, one_hot_labels = preprocess_sample(inputs, negative_labels)
            ff_labels = torch.zeros(len(inputs), 1).to(device)
            loss, output = model(label_and_input, ff_labels, one_hot_labels.detach().requires_grad_())
            # loss.backward() # TODO (should we do the linear regressor step for negative data?)

            # optimizer.step()
            # optimizer.zero_grad()

            # # making the output of the model 1 hot encoding
            # max_idx = torch.argmax(output)
            # output = torch.zeros_like(output)
            # output[max_idx] = 1

            # # update accuracy calcs after positive data step
            # sample_counter += 1
            # epoch_accuracy += is_accurate(label, output)
        print(f"Epoch: {epoch} -- Sample: {sample_counter}/{len(train_loader.dataset)} -- Error rate: {epoch_accuracy}/{sample_counter}({round(epoch_accuracy/sample_counter*100, 2)}%)") # just doing this for positive samples for now
        test(model)
    return model

def test(model):
    wrong = 0
    i = 0
    total = len(test_loader.dataset)
    for inputs, labels, in test_loader:
        i += len(inputs)
        label_and_input, _ = preprocess_sample(inputs, torch.zeros(len(inputs), 10).to(device))
        prediction = model.infer(label_and_input)
        #prediction = model.slower_inference(inputs)
        wrong += absolute_loss(labels, prediction)

    print(f"Error rate: {wrong} / {i} ({round(wrong/i*100, 2)}%)")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def absolute_loss(label, output, verbose=False):
    label = label.to(device)
    output = output.to(device)
    if output[0].shape == torch.Size([10]):
        output = torch.argmax(output, -1)
    if label[0].shape == torch.Size([10]):
        label = torch.argmax(label, -1)

    if verbose:
        diff_mask = label != output
        label_diff_values = label[diff_mask]
        output_diff_values = output[diff_mask]
        print("------------------------------Diffs----------------------------------")
        print(f"Label value: {label_diff_values.tolist()}")
        print(f"  Out value: {output_diff_values.tolist()}")
        
    return torch.count_nonzero(label - output, -1).item()


# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1]
])

print("Downloading the training dataset to ./data")
train_dataset = datasets.MNIST(
    root='./data',  # Directory to store the dataset
    train=True,  # Load the training set
    download=True,  # Download the dataset if not available locally
    transform=transform  # Apply transformations
)

print("Downloading the testing dataset to ./data")
test_dataset = datasets.MNIST(
    root='./data',
    train=False,  # Load the test set
    download=True,
    transform=transform
)

# Create DataLoader objects for batching and shuffling
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=128,  # Number of samples per batch
    shuffle=True  # Shuffle the dataset
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False  # No need to shuffle the test set
)

layers = [794, 2000, 2000, 2000, 2000] # TODO (mess with these values too)
model = FwdFwdModel(layers)
model = model.to(device)
optimizer = torch.optim.Adam(model.get_linear_classifier_param(), lr=0.0001)

trained_model = train(30, model, optimizer, train_loader)

with open("model.pkl", "wb") as file:
    pickle.dump(trained_model, file)