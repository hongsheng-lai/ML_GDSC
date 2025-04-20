import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import logging
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import your dataset and models. Adjust paths as needed.
from sklearn.metrics import mean_squared_error, r2_score
from utils.GDSCDataset import GDSCDataset
from utils.model import MLP

class Trainer:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the Trainer with the provided configuration.
        
        Args:
            cfg: Configuration dictionary loaded via Hydra/OmegaConf. It must include 
                 keys for 'data', 'train', 'test', 'model', 'wandb', and 'general'.
        """
        self.cfg = cfg
        self.setup_logging()
        self.load_device()
        
        # Create directory for saving the model checkpoint
        self.model_save_path = cfg.model.model_save_path
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Load the dataset (assumed to be pre-padded with mask information)
        print("Loading dataset...")
        self.load_data()
        # Initialize the model
        print("Loading model...")
        self.load_model()
        
        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.learning_rate)
    
    def setup_logging(self):
        """Set up basic logging to console."""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_device(self):
        """Set device to GPU if available (with seeding), else CPU."""
        cfg = self.cfg
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cfg.general.gpu_id}")
            torch.manual_seed(cfg.general.seed)
            torch.cuda.manual_seed_all(cfg.general.seed)
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("No GPU detected, using CPU")
    
    def load_data(self):
        """
        Load the dataset and split into train, validation, and test sets 
        using scikit-learn's train_test_split.
        """
        dataset = GDSCDataset()
        total_indices = list(range(len(dataset)))
        cfg_data = self.cfg.data
        
        # Calculate test ratio based on train_split and val_split provided in the config.
        test_ratio = 1 - (cfg_data.train_split + cfg_data.val_split)
        # First, split into train+val and test sets.
        train_val_indices, test_indices = train_test_split(
            total_indices, test_size=test_ratio, random_state=self.cfg.general.seed)
        # Next, split train_val_indices into train and validation sets.
        # The ratio for validation is adjusted relative to the train+val set.
        val_ratio_adjusted = cfg_data.val_split / (cfg_data.train_split + cfg_data.val_split)
        train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_ratio_adjusted, random_state=self.cfg.general.seed)
        
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.model.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.model.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.cfg.model.batch_size, shuffle=False)
        
        self.logger.info(f"Data loaded: {len(self.train_dataset)} train, "
                         f"{len(self.val_dataset)} val, {len(self.test_dataset)} test samples")
    
    def load_model(self):
        """Initialize the model based on the configuration."""
        cfg_model = self.cfg.model
        self.model = MLP(cfg_model).to(self.device)
        self.cfg.model.baseline = False
        self.logger.info(f"Model loaded: {cfg_model.model_choice}")
    
    def load_wandb(self):
        """Initialize wandb logging."""
        cfg = self.cfg
        run_id = cfg.wandb.run_id if cfg.wandb.run_id else wandb.util.generate_id()
        OmegaConf.set_struct(cfg, False)
        cfg.wandb.run_id = run_id
        OmegaConf.set_struct(cfg, True)
        self.wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            id=run_id,
            resume=True
        )
        self.logger.info(f"Initialized wandb with run_id: {run_id}")


    def train(self):
        """
        Train the model using the training set and evaluate on the validation set 
        after each epoch. Implements early stopping based on validation accuracy.
        Metrics are logged to wandb.
        """
        self.load_wandb()
        num_epochs = self.cfg.train.num_epochs
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = self.cfg.train.early_stop_patience

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_r2 = 0
            total_train = 0
            
            # Training loop with tqdm progress bar
            train_bar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)
            all_preds  = []
            all_labels = []

            for embeddings, labels in train_bar:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device).squeeze(1)
                self.optimizer.zero_grad()
                outputs = self.model(embeddings).squeeze(dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * embeddings.size(0)
                total_train += labels.size(0)

                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                train_bar.set_postfix(loss=f"{running_loss/total_train:.4f}")

            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_labels)
            train_r2 = r2_score(y_true, y_pred)
            train_loss = running_loss / total_train
            val_loss, val_r2 = self._evaluate(self.val_loader)

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_r2": train_r2,
                "val_loss": val_loss,
                "val_r2": val_r2
            })

            self.logger.info(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | "
                             f"Train r2: {train_r2:.4f} | Val Loss: {val_loss:.4f} | Val r2: {val_r2:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state_dict": self.model.state_dict()}, self.model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print("Current loss:", val_loss, "Lowest loss:", best_val_loss)
                self.logger.info(f"No improvement in val loss for {patience_counter} epoch(s).")
            
            if patience_counter >= early_stop_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        wandb.finish()

    
    def _evaluate(self, loader):
        """
        Evaluate the model on the provided DataLoader.
        
        Args:
            loader: DataLoader for the evaluation set.
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds  = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device).squeeze(1)
                outputs = self.model(embeddings).squeeze(dim=1)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * embeddings.size(0)
                total += labels.size(0)
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        
        avg_loss = total_loss / len(loader.dataset)
        r2 = r2_score(y_true, y_pred)
        return avg_loss, r2
    
    def evaluate(self):
        """
        Evaluate the best saved model on the test set.
        """
        checkpoint = torch.load(self.model_save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Evaluating model on test set...")
        
        test_loss, test_r2 = self._evaluate(self.test_loader)
        self.logger.info(f"Test Loss: {test_loss:.4f} | Test r2: {test_r2:.4f}")

@hydra.main(version_base=None, config_path="./", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Convert configuration to a mutable container.
    mutable_cfg = OmegaConf.to_container(cfg, resolve=True)
    mutable_cfg = OmegaConf.create(mutable_cfg)

    trainer = Trainer(mutable_cfg)
    if cfg.general.usage == "train":
        trainer.train()
        trainer.evaluate()  # Evaluate on test set after training.
    elif cfg.general.usage == "eval":
        trainer.evaluate()
        pass

if __name__ == "__main__":
    main()
    print("\n=============== No Bug No Error, Finished!!! ===============\n")