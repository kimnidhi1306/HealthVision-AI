# main.py

from data_handling import *
from image_processing import *
from visualization import *
from data_preprocessing import *
from model_training import *
from evaluation import *

# Load and preprocess data
IMAGE_PATH = Path("/content/Data/MCF-7 cell populations Dataset/images")
MASK_PATH = Path("/content/Data/MCF-7 cell populations Dataset/masks")

IMAGE_PATH_LIST = list(IMAGE_PATH.glob("*.png"))
MASK_PATH_LIST = list(MASK_PATH.glob("*.png"))

IMAGE_PATH_LIST = sorted(IMAGE_PATH_LIST)
MASK_PATH_LIST = sorted(MASK_PATH_LIST)

data = pd.DataFrame({'Image': IMAGE_PATH_LIST, 'Mask': MASK_PATH_LIST})

data_train, data_val, data_test = train_val_split(data)

BATCH_SIZE = 4
NUM_WORKERS = os.cpu_count()

train_dataloader, val_dataloader = create_data_loaders(data_train, data_val, BATCH_SIZE, NUM_WORKERS)

# Define and train the model
model = smp.Unet()

freeze_encoder_params(model)

loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

early_stopping = EarlyStopping(patience=10, delta=0.)

EPOCHS = 100
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

RESULTS = train(model.to(device=DEVICE),
                train_dataloader,
                val_dataloader,
                loss_fn,
                optimizer,
                early_stopping,
                EPOCHS)

# Plot training results
loss_and_metric_plot(RESULTS)

# Load test data and evaluate the model
test_dataset = CustomImageMaskDataset(data_test, image_transforms, mask_transforms)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

pred_mask_test = predictions(model, test_dataloader)

IMAGE_TEST = []
MASK_TEST = []

for img, mask in test_dataloader:
    IMAGE_TEST.append(img)
    MASK_TEST.append(mask)

IMAGES_TEST = torch.cat(IMAGE_TEST)
MASKS_TEST = torch.cat(MASK_TEST)

iou_test = calculate_iou(pred_mask_test, MASKS_TEST)
print(f'IOU Test = {iou_test:.4f}')

# Visualize results
show_results(IMAGES_TEST, pred_mask_test, num_images=28)
