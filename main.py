from dataset import get_train_test_loaders
from model_v1 import ResNet
import logging
from utils import dataset_visualizer, identify_optim_lr
import torch
from torchsummary import summary
from model_train_utils import prepare_and_train
from config import Config

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format=Config.LOG_FORMAT,
    force=True  # <-- important in Jupyter
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Hello from Imagenette trainer!")

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    logger.info(f"Cuda available = {is_cuda}, Using device = {device}")

    data_set_path = Config.DATA_SET_PATH
    train_loader, val_loader = get_train_test_loaders(data_set_path, batch_size=Config.BATCH_SIZE, image_size=Config.IMAGE_SIZE, norm_mean=Config.NORM_MEAN, norm_std=Config.NORM_STD)
    
    dataset_visualizer(train_loader)

    model = ResNet(layers=Config.RESNET_LAYERS, num_classes=Config.NUM_CLASSES, use_depthwise=Config.USE_DEPTHWISE).to(device)

    dummy_data = torch.randn(5, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    dummy_output = model(dummy_data)
    logger.info(f"Output shape: {dummy_output.shape}")  # should be [5, Config.NUM_CLASSES]
    summary(model, input_size=(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), device=str(device))
    
    lr_from_lr_finder = Config.LEARNING_RATE
    if str(device) == 'cuda':
        suggested_lr = identify_optim_lr(model, device, train_loader, Config.LR_FINDER_END_LR, Config.LR_FINDER_NUM_ITER)
        if suggested_lr:
            lr_from_lr_finder = suggested_lr
            logger.info(f"Using learning rate from LR finder: {lr_from_lr_finder}")

    logger.info(f"Using learning rate: {lr_from_lr_finder}, Type of lr = {type(lr_from_lr_finder)}")

    prepare_and_train(model, lr_from_lr_finder, device, train_loader, val_loader, total_epochs=Config.TOTAL_EPOCHS)

if __name__ == '__main__':
    main()