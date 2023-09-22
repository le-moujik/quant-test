import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.ensemble import RandomForestClassifier


class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, 
                               kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, 
                               kernel_size=5)
        self.fc1 = nn.Linear(18 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 18 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class DigitClassificationInterface():
    def train(self, train_data):
        raise NotImplementedError()
        
    def _get_classifier(self):
        pass
    
    def _process_data(self, inp):
        pass
    
    def predict(self, inp):
        pass


class RandomModel(DigitClassificationInterface):
    def _process_data(self, inp: torch.Tensor) -> np.array:
        return inp.numpy()[9:19, 9:19]
    
    def predict(self, inp: torch.Tensor) -> int:
        inp = self._process_data(inp)
        return int(np.sum(inp) % 9)


class RFModel(DigitClassificationInterface):
    def __init__(self):
        self.classifier = self._get_classifier()
        
    def _get_classifier(self) -> RandomForestClassifier:
        # we can-t use predict() with untrained classifier
        classifier = RandomForestClassifier()
        train_x = np.random.randint(0, 225, size=(20, 784))
        train_y = np.random.randint(1, 10, size=20)
        classifier.fit(train_x, train_y)
        # model = joblib.load(RF_MODEL_PATH)
        return classifier
    
    def _process_data(self, inp: torch.Tensor) -> np.array:
        return inp.ravel().view(1, -1).numpy()
    
    def predict(self, inp: torch.Tensor) -> int:
        inp = self._process_data(inp)
        result = self.classifier.predict(inp)
        return int(result[0])


class CNNModel(DigitClassificationInterface):
    def __init__(self):
        self.classifier = self._get_classifier()
        
    def _get_classifier(self) -> PyTorchCNN:
        classifier = PyTorchCNN()
        # model.load_state_dict(torch.load(CNN_WEIGHTS_PATH))
        classifier.eval()
        return classifier

    def _process_data(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.to(dtype=torch.float)
    
    def predict(self, inp) -> int:
        inp = self._process_data(inp)
        with torch.no_grad():
            result = self.classifier(inp)
        _, result = torch.max(result, 1)
        # return labels_list[result]
        return int(result+1) # let's say the labels are sorted


class DigitClassifier():
    def __init__(self, alg):
        self.dct = {"rand": RandomModel,
                    "rf": RFModel,
                    "cnn": CNNModel}
        self.model = self.dct[alg]()
    def predict(self, inp) -> int:
        return self.model.predict(inp)


def main():
    mnist = datasets.MNIST(root="MNIST", download=True,
                           train = False, transform=ToTensor())
    mnist = mnist.data[:10]

    rnd_clf = DigitClassifier("rand")
    rf_clf = DigitClassifier("rf")
    cnn_clf = DigitClassifier("cnn")

    for pic in mnist:
        # reshape original (1, 28, 28) to (28, 28, 1) from the task paper
        pic = pic.view(28, 28, 1)
        rnd_res = rnd_clf.predict(pic)
        rf_res = rf_clf.predict(pic)
        cnn_res = cnn_clf.predict(pic)
        print(f"random result: {rnd_res}, "
              f"RF result: {rf_res}, CNN result: {cnn_res}")


main()
