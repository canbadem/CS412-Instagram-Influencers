import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

import common

print("=====================================")
print("Regression KNN")
print("=====================================")

#Set up vectorizer using features=5000 and simple posts aggregator
x_post_train, y_train, x_post_test, df_tfidf = common.VectorizeDataset(5000, common.UserAggregatorPosts)
x_train, x_val, y_train, y_val = train_test_split(df_tfidf, y_train, test_size=0.2, stratify=y_train)

def extract_features_from_posts(posts):
    """Extract and aggregate features from a user's posts with enhanced feature engineering."""
    if not posts:
        return None
    
    # Initialize feature aggregations
    total_comments = []
    media_type_counts = {'IMAGE': 0, 'VIDEO': 0, 'CAROUSEL_ALBUM': 0}
    caption_lengths = []
    like_counts = []
    post_hours = []
    engagement_rates = []  # New feature
    
    for post in posts:
        # Comments and likes
        comments = post.get('comments_count', 0)
        likes = post.get('like_count', 0)
        
        if comments is not None and likes is not None:
            total_comments.append(comments)
            like_counts.append(likes)
            # Engagement rate = (likes + comments) per post
            engagement_rates.append((likes + comments))
            
        # Media type
        media_type = post.get('media_type', 'IMAGE')
        if media_type in media_type_counts:
            media_type_counts[media_type] += 1
            
        # Caption length
        caption = post.get('caption', '')
        if caption is not None:
            caption_lengths.append(len(caption))
            
        # Hour of posting
        timestamp = post.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                # Convert hour to cyclic features
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                post_hours.append((hour_sin, hour_cos))
            except (ValueError, TypeError):
                pass
    
    # Compute advanced aggregated features
    recent_posts = posts[-5:] if len(posts) >= 5 else posts  # Last 5 posts
    recent_likes = [post.get('like_count', 0) for post in recent_posts if post.get('like_count') is not None]
    
    features = [
        np.mean(like_counts) if like_counts else 0,  # Average likes
        np.median(like_counts) if like_counts else 0,  # Median likes
        np.percentile(like_counts, 75) if like_counts else 0,  # 75th percentile
        np.mean(recent_likes) if recent_likes else 0,  # Recent average likes
        np.mean(total_comments) if total_comments else 0,  # Average comments
        np.median(total_comments) if total_comments else 0,  # Median comments
        np.mean(caption_lengths) if caption_lengths else 0,  # Average caption length
        np.std(caption_lengths) if len(caption_lengths) > 1 else 0,  # Caption length variance
        media_type_counts['IMAGE'] / len(posts) if posts else 0,
        media_type_counts['VIDEO'] / len(posts) if posts else 0,
        media_type_counts['CAROUSEL_ALBUM'] / len(posts) if posts else 0,
        np.mean([h[0] for h in post_hours]) if post_hours else 0,  # Hour sine
        np.mean([h[1] for h in post_hours]) if post_hours else 0,  # Hour cosine
        np.std(like_counts) if len(like_counts) > 1 else 0,  # Like count variance
        np.std(total_comments) if len(total_comments) > 1 else 0,  # Comment variance
        len(posts),  # Total posts
        np.mean(engagement_rates) if engagement_rates else 0,  # Average engagement
        np.max(like_counts) if like_counts else 0,  # Maximum likes
        np.min(like_counts) if like_counts else 0,  # Minimum likes
    ]
    
    return features

# Step 1: Prepare training data
print("Preparing training data")
user_features = []
like_counts = []

for uname, posts in common.username2posts_train.items():
    features = extract_features_from_posts(posts)
    if features is not None:
        user_features.append(features)
        valid_like_counts = [post.get("like_count", 0) for post in posts if post.get("like_count") is not None]
        like_counts.append(np.mean(valid_like_counts) if valid_like_counts else 0)

user_features = np.array(user_features)
like_counts = np.array(like_counts)

# Step 2: Scale features using RobustScaler (better for outliers)
print("Fitting scaler")
scaler = RobustScaler()
user_features_scaled = scaler.fit_transform(user_features)

# Step 3: Find optimal KNN parameters
print("Finding optimal KNN parameters using grid search")
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'p': [1, 2]
}

grid_search = GridSearchCV(
    KNeighborsRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=10
)

grid_search.fit(user_features_scaled, like_counts)
best_knn = grid_search.best_estimator_

print("Best parameters and KNN are found")
print(f"Best KNN parameters: {grid_search.best_params_}")

def predict_like_count_knn(username, current_post=None):
    if username in common.username2posts_train:
        posts = common.username2posts_train[username]
    elif username in common.username2posts_test:
        posts = common.username2posts_test[username]
    else:
        return 0
    
    features = extract_features_from_posts(posts)
    if features is None:
        return 0
        
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Get KNN prediction
    knn_pred = best_knn.predict(features_scaled)[0]
    
    # Combine with simple average for more robust prediction
    valid_like_counts = [post.get("like_count", 0) for post in posts if post.get("like_count") is not None]
    avg_pred = np.mean(valid_like_counts) if valid_like_counts else 0
    
    # Weighted ensemble of KNN and average prediction
    final_pred = 0.7 * avg_pred + 0.3 * knn_pred  # Adjust weights based on validation
    return max(0, final_pred)

def log_mse_like_counts(y_true, y_pred):
  """
  Calculate the Log Mean Squared Error (Log MSE) for like counts (log(like_count + 1)).

  Parameters:
  - y_true: array-like, actual like counts
  - y_pred: array-like, predicted like counts

  Returns:
  - log_mse: float, Log Mean Squared Error
  """
  # Ensure inputs are numpy arrays
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  # Log transformation: log(like_count + 1)
  log_y_true = np.log1p(y_true)
  log_y_pred = np.log1p(y_pred)

  # Compute squared errors
  squared_errors = (log_y_true - log_y_pred) ** 2

  # Return the mean of squared errors
  return np.mean(squared_errors)

# Step 4: Evaluate on the train dataset
print("Evaluating on the training dataset")

y_like_count_train_true = []
y_like_count_train_pred_knn = []

for uname, posts in common.username2posts_train.items():
    for post in posts:
        true_val = post.get("like_count", 0)
        if true_val is None:
            true_val = 0

        pred_val = predict_like_count_knn(uname, post)

        y_like_count_train_true.append(true_val)
        y_like_count_train_pred_knn.append(pred_val)

print(f"Log MSE Train (KNN)= {log_mse_like_counts(y_like_count_train_true, y_like_count_train_pred_knn)}")

# Step 5: Predict on test dataset and create prediction dictionary
predictions_dict = {}

with open(common.test_regression_path, "rt") as fh:
    for line in fh:
        sample = json.loads(line)
        post_id = sample["id"]  # Assuming post_id exists in the input data
        pred_val = predict_like_count_knn(sample["username"])
        predictions_dict[post_id] = int(pred_val)  # Convert prediction to integer

# Save predictions to JSON file
output_filename = "prediction-regression-round1.json"  # Adjust round number as needed
with open(output_filename, "w") as f:
    json.dump(predictions_dict, f)

print(f"Predictions written to {output_filename}")