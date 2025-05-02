import argparse
import torch
from data_setup import get_data_loaders
from model_builder import SignatureCNN
from engine import train
from utils import load_model, get_preds, compute_classification_metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
        args.train_dir, args.test_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    model = SignatureCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, test_loader, criterion, optimizer,
          args.epochs, device, args.save_path)

    y_true, y_pred = get_preds(model, test_loader, device)
    compute_classification_metrics(y_true, y_pred, train_dataset.classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signature Verification Training")
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir',  type=str, required=True)
    parser.add_argument('--img_size',   nargs=2, type=int, default=[224,224])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--epochs',     type=int, default=40)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--save_path',  type=str, default='best_signature_model.pth')
    args = parser.parse_args()
    main(args)
