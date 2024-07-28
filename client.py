from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch 
import flwr as fl
from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloaders, valloaders, num_classes)-> None:
        super().__init__()

        self.trainloaders = trainloaders
        self.valloaders= valloaders
        self.model = Net(num_classes)
        self.device = torch.device("cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state.dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config['lr']
        momentum = config['momentum']
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optim, epochs, self.device)

        
        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {}
    
def generate_client_fn(trainloaders, valloaders, input_size, output_size):
    """Return a function that can be used by the VirtualClientEngine to spawn a FlowerClient with client id `cid`."""

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FlowerClient that will use the cid-th train/val
        # dataloaders as its local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            input_size=input_size,
            output_size=output_size,
        ).to_client()

    # return the function to spawn client
    return client_fn

