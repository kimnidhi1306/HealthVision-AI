# evaluation.py

from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

def predictions(model, test_dataloader):
    model.eval()
    pred_mask_test = []

    with torch.no_grad():
        for X, _ in tqdm(test_dataloader):
            X = X.to(device=DEVICE, dtype=torch.float32)
            logit_mask = model(X)
            prob_mask = logit_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            pred_mask_test.append(pred_mask.detach().cpu())

    pred_mask_test = torch.cat(pred_mask_test)

    return pred_mask_test

def calculate_iou(pred_mask_test, MASKS_TEST):
    TP, FP, FN, TN = smp.metrics.get_stats(pred_mask_test.long(),
                                           (MASKS_TEST > 0.5).float().long(),
                                           mode="binary")
    iou_test = smp.metrics.iou_score(TP, FP, FN, TN, reduction="micro")
    return iou_test

def show_results(IMAGES_TEST, pred_mask_test, num_images=28):
    fig, ax = plt.subplots(nrows=num_images, ncols=2, figsize=(15, 60))
    
    for i, (img_tst, pred_mk) in enumerate(zip(IMAGES_TEST, pred_mask_test)):
        ax[i, 0].imshow(img_tst.permute(1, 2, 0).numpy())
        ax[i, 0].set_title("Original Image", fontsize=10)
        ax[i, 0].axis('off')
        
        ax[i, 1].imshow(pred_mk.squeeze().numpy())
        ax[i, 1].set_title("Predicted Mask", fontsize=10)
        ax[i, 1].axis('off')

    fig.show()
