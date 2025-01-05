import pickle
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
            self.optimizers.append(torch.optim.Adam(self.model[len(self.model) - 1].parameters()))

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

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)
        logits = sum_of_squares - z.shape[-1] # TODO (not sure if authors used size of layer as threshold)
        logits = torch.reshape(torch.sigmoid(logits), (len(z), 1))
        ff_loss = self.ff_loss(logits, labels)

        return ff_loss

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
        return classification_loss, torch.argmax(output, dim=-1)

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
            ff_loss = self._calc_ff_loss(z, ff_labels) # calc layer wise loss
            self.optimizers[optim_idx].zero_grad()

            ff_loss.backward(retain_graph=True) # compute gradients for layer

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
            return output - torch.max(output, dim=-1, keepdim=True)[0]

def preprocess_sample(inputs, labels):
    if labels[0].shape != torch.Size([10]):
        one_hot_labels = F.one_hot(labels, num_classes=10)
    else:
        one_hot_labels = labels
    inputs = torch.reshape(inputs, (len(inputs), 784))
    # concatinate the label as a 1 hot encoding and flattened image
    return torch.cat((one_hot_labels, inputs), -1).to(device), one_hot_labels.to(device)

def train(epochs, model, optimizer, train_loader): # this optimizer and loss is for the linear classifier

    for epoch in range(epochs):
        epoch_accuracy = 0
        sample_counter = 0 # for now ill include negative samples in this too
        for inputs, labels, in train_loader:

            # positive data forward
            optimizer.zero_grad()

            # making the label 1 hot encoding
            label_and_input, one_hot_labels = preprocess_sample(inputs, labels)
            ff_labels = torch.ones(len(inputs), 1).to(device)
            loss, output = model(label_and_input, ff_labels, one_hot_labels) # TODO (not sure if using model output -> neg data pipeline is a good idea anymore)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # negative data forward
            # optimizer.zero_grad()

            # update accuracy calcs after positive data step
            sample_counter += len(inputs)
            epoch_accuracy += absolute_loss(torch.argmax(one_hot_labels, -1), output)
            label_and_input, one_hot_labels = preprocess_sample(inputs, output)
            ff_labels = torch.zeros(len(inputs), 1).to(device)
            breakpoint()
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
            if sample_counter % 1000 == 0:
                print(f"Epoch: {epoch} -- Sample: {sample_counter}/{len(train_loader.dataset)} -- Accuracy rate: {epoch_accuracy}/{sample_counter}({round(epoch_accuracy/sample_counter*100, 2)}%)") # just doing this for positive samples for now
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
        wrong += absolute_loss(labels, prediction)
    print(f"Error rate: {wrong} / {i} ({round(wrong/i*100, 2)}%)")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def absolute_loss(label, output):
    return torch.count_nonzero(label - output, -1).item()


# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1]
])

print("Downloading the training dataset to ./data")
train_dataset = datasets.FashionMNIST(
    root='./data',  # Directory to store the dataset
    train=True,  # Load the training set
    download=True,  # Download the dataset if not available locally
    transform=transform  # Apply transformations
)

print("Downloading the testing dataset to ./data")
test_dataset = datasets.FashionMNIST(
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

# Example: Iterate through the training DataLoader
for image, label in train_loader:
    image = image.squeeze(0)  # Remove the batch dimension, making it [1, 28, 28]
    #label = label.item()  # Convert the label tensor to a Python integer
    print(f"Image shape: {image.shape}, Label: {label.shape}")
    break


layers = [794, 2000, 2000, 2000, 2000]
model = FwdFwdModel(layers)
model = model.to(device)
optimizer = torch.optim.Adam(model.get_linear_classifier_param())

trained_model = train(30, model, optimizer, train_loader)

with open("model.pkl", "wb") as file:
    pickle.dump(trained_model, file)