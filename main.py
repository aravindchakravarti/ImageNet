from dataset import get_train_test_loaders
from model_v1 import ResNet
import logging
from utils import show_dataset_stats, dataset_visualizer, indentify_optim_lr
import torch
from torchsummary import summary
from model_train_utils import prepare_and_train



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # <-- important in Jupyter
)
logger = logging.getLogger(__name__)

import sys
sys.path.append('..')


def main():
    logger.info("Hello from Imagenette trainer!")

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    logger.info(f"Cuda available = {is_cuda}, Using device = {device}")

    data_set_path = 'C:/Users/achakravarti/Documents/ERAv4/imagenette2-320'
    train_loader, val_loader = get_train_test_loaders(data_set_path)
    # show_dataset_stats(train_loader)
    dataset_visualizer(train_loader)

    model = ResNet(layers=[2,2,3,2], num_classes=10, use_depthwise=(False, False, True, True)).to(device)

    dummy_data = torch.randn(5, 3, 224, 224).to(device)
    dummy_output = model(dummy_data)
    logger.info("Output shape:", dummy_output.shape)  # should be [5, 100]
    summary(model, input_size=(3, 224, 224), device=str(device))
    
    if device == 'cuda':
        indentify_optim_lr(device, train_loader)

    lr_from_lr_finder = input("Enter lr from lr finder:")
    logger.info("Going to use lr from lr finder:", lr_from_lr_finder)

    model = ResNet(layers=[2, 2, 3, 2], num_classes=10,
               use_depthwise=(False, False, True, True)).to(device)

    
    prepare_and_train(model, lr_from_lr_finder, device, train_loader, val_loader, total_epochs=15)



if __name__ == '__main__':
    main()


