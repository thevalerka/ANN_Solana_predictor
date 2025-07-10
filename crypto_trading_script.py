import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
DATA_PATH = "/home/ubuntu/009_MM_BOTS/bot004_NeuralNetwork/data/balance_change_records.json"
MODEL_SAVE_DIR = "/home/ubuntu/009_MM_BOTS/bot004_NeuralNetwork/saved_models/"

def load_data_from_file(file_path):
    """
    Load cryptocurrency data from JSON file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

def save_model_components(model, scaler, feature_columns, label_encoder, save_dir):
    """
    Save all model components for later use
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the neural network model
    model_path = os.path.join(save_dir, "crypto_trading_model.h5")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature columns
    features_path = os.path.join(save_dir, "feature_columns.pkl")
    with open(features_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"Feature columns saved to: {features_path}")
    
    # Save label encoder
    encoder_path = os.path.join(save_dir, "label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to: {encoder_path}")
    
    # Save model metadata
    metadata = {
        'model_version': '1.0',
        'training_date': datetime.now().isoformat(),
        'feature_count': len(feature_columns),
        'target_classes': ['SHORT', 'HOLD', 'LONG']
    }
    
    metadata_path = os.path.join(save_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")

def load_model_components(save_dir):
    """
    Load all saved model components
    """
    try:
        # Load the neural network model
        model_path = os.path.join(save_dir, "crypto_trading_model.h5")
        model = load_model(model_path)
        
        # Load the scaler
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature columns
        features_path = os.path.join(save_dir, "feature_columns.pkl")
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Load label encoder
        encoder_path = os.path.join(save_dir, "label_encoder.pkl")
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(save_dir, "model_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("All model components loaded successfully!")
        print(f"Model trained on: {metadata['training_date']}")
        print(f"Feature count: {metadata['feature_count']}")
        
        return model, scaler, feature_columns, label_encoder, metadata
        
    except Exception as e:
        print(f"Error loading model components: {e}")
        return None, None, None, None, None

# Load the actual data from file
raw_data = load_data_from_file(DATA_PATH)


def safe_datetime_parse(time_series):
    """
    Safely parse datetime strings with multiple possible formats
    """
    try:
        # Try ISO8601 format first
        return pd.to_datetime(time_series, format='ISO8601')
    except:
        try:
            # Try mixed format
            return pd.to_datetime(time_series, format='mixed')
        except:
            try:
                # Try infer_datetime_format
                return pd.to_datetime(time_series, infer_datetime_format=True)
            except:
                # Last resort - let pandas figure it out
                return pd.to_datetime(time_series)

def preprocess_crypto_data(raw_data):
    """
    Preprocess the cryptocurrency data for neural network training
    """
    if raw_data is None:
        print("No data to process!")
        return None, None
    
    print("Processing cryptocurrency data...")
    print(f"Raw data contains {len(raw_data)} records")
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    
    # Check timestamp formats
    sample_times = df['time'].head(3).tolist()
    print(f"Sample timestamps: {sample_times}")
    
    # Convert time to datetime with safe parsing
    df['time'] = safe_datetime_parse(df['time'])
    df = df.sort_values(['coinname', 'time']).reset_index(drop=True)
    
    print(f"Successfully parsed {len(df)} timestamps")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"Unique coins: {df['coinname'].unique().tolist()}")
    
    # Encode categorical variables
    le = LabelEncoder()
    df['change_direction_encoded'] = le.fit_transform(df['change_direction'])
    
    # Feature engineering for each coin
    processed_data = []
    
    for coin in df['coinname'].unique():
        coin_data = df[df['coinname'] == coin].copy().reset_index(drop=True)
        
        if len(coin_data) < 3:  # Skip coins with insufficient data
            continue
            
        # Price-based features
        coin_data['price_change'] = coin_data['price_of_coin'].pct_change()
        coin_data['price_ma_2'] = coin_data['price_of_coin'].rolling(window=2).mean()
        coin_data['price_ma_3'] = coin_data['price_of_coin'].rolling(window=3).mean()
        coin_data['price_std_2'] = coin_data['price_of_coin'].rolling(window=2).std()
        
        # Volume/Amount features
        coin_data['amount_ma_2'] = coin_data['amount'].rolling(window=2).mean()
        coin_data['amount_std_2'] = coin_data['amount'].rolling(window=2).std()
        coin_data['percent_value_ma_2'] = coin_data['percent_value'].rolling(window=2).mean()
        coin_data['percent_value_std_2'] = coin_data['percent_value'].rolling(window=2).std()
        
        # Enhanced momentum features
        coin_data['momentum'] = coin_data['price_of_coin'] / coin_data['price_ma_2']
        coin_data['momentum_3'] = coin_data['price_of_coin'] / coin_data['price_ma_3']
        coin_data['consecutive_momentum'] = coin_data['consecutive_count'] * coin_data['percent_value']
        coin_data['amount_momentum'] = coin_data['amount'] / (coin_data['amount_ma_2'] + 1e-8)
        
        # Direction consistency features
        coin_data['direction_encoded'] = coin_data['change_direction_encoded']
        coin_data['direction_change'] = coin_data['change_direction_encoded'].diff().fillna(0)
        
        # Price acceleration
        coin_data['price_acceleration'] = coin_data['price_change'].diff()
        
        # Time-based features
        coin_data['hour'] = coin_data['time'].dt.hour
        coin_data['minute'] = coin_data['time'].dt.minute
        coin_data['time_since_start'] = (coin_data['time'] - coin_data['time'].min()).dt.total_seconds() / 3600  # hours
        
        # Volatility and risk features
        coin_data['volatility'] = coin_data['price_change'].rolling(window=2).std()
        coin_data['volume_volatility'] = coin_data['amount'].rolling(window=2).std()
        coin_data['percent_volatility'] = coin_data['percent_value'].rolling(window=2).std()
        
        # Create target variable (future profit potential)
        # Look ahead to determine if price will increase or decrease
        future_returns_1 = coin_data['price_of_coin'].shift(-1) / coin_data['price_of_coin'] - 1
        future_returns_2 = coin_data['price_of_coin'].shift(-2) / coin_data['price_of_coin'] - 1
        
        # Weighted future return
        weighted_future_return = (0.7 * future_returns_1 + 0.3 * future_returns_2)
        
        # Analyze return distribution to set better thresholds
        return_std = weighted_future_return.std()
        return_mean = weighted_future_return.mean()
        
        # Define adaptive thresholds based on data distribution
        long_threshold = max(0.005, return_mean + 0.5 * return_std)   # At least 0.5% or data-driven
        short_threshold = min(-0.005, return_mean - 0.5 * return_std)  # At least -0.5% or data-driven
        
        print(f"   {coin}: Return std={return_std:.4f}, mean={return_mean:.4f}")
        print(f"   {coin}: LONG threshold={long_threshold:.4f}, SHORT threshold={short_threshold:.4f}")
        
        # Create target labels with more balanced distribution
        target = np.where(weighted_future_return > long_threshold, 2,    # LONG
                         np.where(weighted_future_return < short_threshold, 0,  # SHORT
                                 1))  # HOLD
        
        # Check target distribution for this coin
        target_dist = pd.Series(target).value_counts().sort_index()
        print(f"   {coin}: Target distribution - SHORT:{target_dist.get(0,0)}, HOLD:{target_dist.get(1,0)}, LONG:{target_dist.get(2,0)}")
        
        coin_data['target'] = target
        coin_data['future_return'] = weighted_future_return
        
        processed_data.append(coin_data)
    
    # Combine all coins
    final_df = pd.concat(processed_data, ignore_index=True)
    
    # Remove rows with NaN values
    final_df = final_df.dropna().reset_index(drop=True)
    
    print(f"Processed data shape: {final_df.shape}")
    print(f"Target distribution:")
    print(final_df['target'].value_counts().sort_index())
    
    return final_df, le

def build_neural_network(input_shape):
    """
    Build a neural network for crypto trading decisions with improved architecture
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        Dense(8, activation='relu'),
        Dropout(0.2),
        
        Dense(3, activation='softmax')  # 3 classes: SHORT(0), HOLD(1), LONG(2)
    ])
    
    # Use class weights to handle imbalanced data
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_crypto_model():
    """
    Train the cryptocurrency trading model
    """
    print("=== Cryptocurrency Trading Neural Network ===\n")
    
    # Check if data was loaded successfully
    if raw_data is None:
        print("Failed to load data. Exiting...")
        return None, None, None, None, None
    
    # Preprocess data
    df, label_encoder = preprocess_crypto_data(raw_data)
    
    if df is None:
        print("Failed to preprocess data. Exiting...")
        return None, None, None, None, None
    
    # Prepare features
    feature_columns = [
        'change_direction_encoded', 'consecutive_count', 'amount', 'percent_value',
        'price_of_coin', 'price_change', 'price_ma_2', 'price_ma_3', 'price_std_2',
        'amount_ma_2', 'amount_std_2', 'percent_value_ma_2', 'percent_value_std_2',
        'momentum', 'momentum_3', 'consecutive_momentum', 'amount_momentum',
        'direction_encoded', 'direction_change', 'price_acceleration',
        'hour', 'minute', 'time_since_start', 'volatility', 'volume_volatility', 'percent_volatility'
    ]
    
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    
    # Detailed target analysis
    print(f"\nTarget Distribution Analysis:")
    target_counts = pd.Series(y).value_counts().sort_index()
    total = len(y)
    print(f"SHORT (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/total*100:.1f}%)")
    print(f"HOLD (1):  {target_counts.get(1, 0)} ({target_counts.get(1, 0)/total*100:.1f}%)")
    print(f"LONG (2):  {target_counts.get(2, 0)} ({target_counts.get(2, 0)/total*100:.1f}%)")
    
    # Check if we have enough data
    if len(X) < 10:
        print("Insufficient data for training. Need at least 10 samples.")
        return None, None, None, None, None
    
    # Check for class imbalance
    min_class_count = target_counts.min() if len(target_counts) > 0 else 0
    if min_class_count == 0:
        print("⚠️ Warning: Some target classes have no samples!")
        print("Adjusting thresholds or feature engineering needed.")
    
    # Calculate class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    
    try:
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y), 
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        print(f"Class weights: {class_weight_dict}")
    except:
        class_weight_dict = None
        print("Could not compute class weights - using equal weights")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (adjust test_size if dataset is small)
    test_size = min(0.3, max(0.1, len(X) // 5))  # Adaptive test size
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        # If stratification fails due to small dataset, split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Build and train model
    model = build_neural_network(X.shape[1])
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Adjust batch size for small datasets
    batch_size = min(16, max(1, len(X_train) // 4))
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,  # Increased epochs
        batch_size=batch_size,
        class_weight=class_weight_dict,  # Use class weights
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Debug prediction distribution
    print(f"\nPrediction Analysis:")
    pred_counts = pd.Series(y_pred_classes).value_counts().sort_index()
    pred_total = len(y_pred_classes)
    print(f"Predicted SHORT (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/pred_total*100:.1f}%)")
    print(f"Predicted HOLD (1):  {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/pred_total*100:.1f}%)")
    print(f"Predicted LONG (2):  {pred_counts.get(2, 0)} ({pred_counts.get(2, 0)/pred_total*100:.1f}%)")
    
    # Show prediction confidence distribution
    print(f"\nPrediction Confidence:")
    confidences = np.max(y_pred, axis=1)
    print(f"Average confidence: {confidences.mean():.4f}")
    print(f"Confidence range: {confidences.min():.4f} - {confidences.max():.4f}")
    
    # Show some sample predictions with probabilities
    print(f"\nSample Predictions (first 5):")
    for i in range(min(5, len(y_pred))):
        probs = y_pred[i]
        predicted = y_pred_classes[i]
        actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        print(f"  Sample {i+1}: Pred={predicted}, Actual={actual}, Probs=[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]")
    
    print("\nClassification Report:")
    target_names = ['SHORT', 'HOLD', 'LONG']
    print(classification_report(y_test, y_pred_classes, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Save model components
    print("\n=== Saving Model Components ===")
    save_model_components(model, scaler, feature_columns, label_encoder, MODEL_SAVE_DIR)
    
    # Backtest strategy
    print("\n=== Backtesting Strategy ===")
    test_df = df.iloc[-len(y_test):].copy()
    test_df['predicted_action'] = y_pred_classes
    
    profits = []
    for idx, row in test_df.iterrows():
        predicted_action = row['predicted_action']
        actual_return = row['future_return']
        
        if predicted_action == 2:  # LONG
            profit = actual_return
        elif predicted_action == 0:  # SHORT
            profit = -actual_return
        else:  # HOLD
            profit = 0
        
        profits.append(profit)
    
    test_df['profit'] = profits
    
    # Calculate performance metrics
    total_return = np.sum(profits)
    win_rate = len([p for p in profits if p > 0]) / len(profits)
    avg_profit_per_trade = np.mean(profits)
    max_profit = np.max(profits)
    min_profit = np.min(profits)
    
    print(f"Backtest Results:")
    print(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
    print(f"Average Profit per Trade: {avg_profit_per_trade:.4f}")
    print(f"Best Trade: {max_profit:.4f}")
    print(f"Worst Trade: {min_profit:.4f}")
    print(f"Number of Trades: {len(profits)}")
    
    # Trading signal distribution
    signal_dist = test_df['predicted_action'].value_counts().sort_index()
    print(f"\nTrading Signal Distribution:")
    for i, count in signal_dist.items():
        action = target_names[i]
        print(f"{action}: {count} ({count/len(test_df)*100:.1f}%)")
    
    return model, scaler, feature_columns, label_encoder, df

def predict_trading_action(model, scaler, feature_columns, new_data):
    """
    Make trading predictions on new data
    
    Args:
        model: Trained neural network model
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
        new_data: Dictionary with feature values
    
    Returns:
        action: 'LONG', 'SHORT', or 'HOLD'
        confidence: Probability of prediction
    """
    # Convert to feature array
    feature_array = np.array([new_data[col] for col in feature_columns])
    feature_array = feature_array.reshape(1, -1)
    
    # Scale features
    feature_scaled = scaler.transform(feature_array)
    
    # Predict
    prediction = model.predict(feature_scaled, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    # Convert to action
    actions = ['SHORT', 'HOLD', 'LONG']
    action = actions[predicted_class]
    
    return action, confidence

# Run the training
if __name__ == "__main__":
    # Train the model
    trained_model, fitted_scaler, features, processed_df = train_crypto_model()
    
    # Example prediction for new data
    print("\n=== Example Trading Prediction ===")
    
    # Use the last row as an example
    last_row = processed_df.iloc[-1]
    sample_data = {col: last_row[col] for col in features}
    
    action, confidence = predict_trading_action(
        trained_model, fitted_scaler, features, sample_data
    )
    
    print(f"Sample Data: {last_row['coinname']} at {last_row['time']}")
    print(f"Price: ${last_row['price_of_coin']:.4f}")
    print(f"Recent Change: {last_row['change_direction']}")
    print(f"Predicted Action: {action}")
    print(f"Confidence: {confidence:.4f}")
    
    print("\n=== Model Training Complete ===")
    print("You can now use the trained model to make trading decisions!")
