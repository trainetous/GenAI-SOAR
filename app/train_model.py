# train_model.py
import pandas as pd
import numpy as np
from pycaret.classification import setup as classification_setup, compare_models, finalize_model, save_model, plot_model as plot_classification_model
from pycaret.clustering import setup as clustering_setup, create_model, predict_model as predict_clustering_model, save_model as save_clustering_model
import os
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples=600): # Increased samples for better clustering
    """Generates a synthetic dataset of phishing and benign URL features with threat actor profiles."""
    print("Generating synthetic dataset...")

    features = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
        'SFH', 'Abnormal_URL', 'has_political_keyword' # New feature
    ]

    num_phishing = num_samples // 2
    num_benign = num_samples - num_phishing

    # Divide phishing samples equally among the three profiles
    num_state_sponsored = num_phishing // 3
    num_organized_cybercrime = num_phishing // 3
    num_hacktivist = num_phishing - num_state_sponsored - num_organized_cybercrime

    # --- State-Sponsored Profile (High sophistication, valid SSL, deceptive prefix/suffix) ---
    state_sponsored_data = {
        'having_IP_Address': np.random.choice([1, -1], num_state_sponsored, p=[0.05, 0.95]),
        'URL_Length': np.random.choice([1, 0, -1], num_state_sponsored, p=[0.2, 0.6, 0.2]), # Normal/Long preferred
        'Shortining_Service': np.random.choice([1, -1], num_state_sponsored, p=[0.05, 0.95]),
        'having_At_Symbol': np.random.choice([1, -1], num_state_sponsored, p=[0.05, 0.95]),
        'double_slash_redirecting': np.random.choice([1, -1], num_state_sponsored, p=[0.05, 0.95]),
        'Prefix_Suffix': np.random.choice([1, -1], num_state_sponsored, p=[0.8, 0.2]), # High prefix/suffix use
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_state_sponsored, p=[0.3, 0.4, 0.3]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_state_sponsored, p=[0.05, 0.15, 0.8]), # High valid SSL
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_state_sponsored, p=[0.1, 0.2, 0.7]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_state_sponsored, p=[0.1, 0.2, 0.7]),
        'SFH': np.random.choice([-1, 0, 1], num_state_sponsored, p=[0.1, 0.1, 0.8]),
        'Abnormal_URL': np.random.choice([1, -1], num_state_sponsored, p=[0.1, 0.9]),
        'has_political_keyword': np.random.choice([1, -1], num_state_sponsored, p=[0.01, 0.99]) # Not typical for state-sponsored
    }
    df_state_sponsored = pd.DataFrame(state_sponsored_data)
    df_state_sponsored['threat_actor_profile'] = 'State-Sponsored'

    # --- Organized Cybercrime Profile (High volume, noisy, shortened, IP, abnormal) ---
    organized_cybercrime_data = {
        'having_IP_Address': np.random.choice([1, -1], num_organized_cybercrime, p=[0.7, 0.3]), # High IP use
        'URL_Length': np.random.choice([1, 0, -1], num_organized_cybercrime, p=[0.4, 0.4, 0.2]),
        'Shortining_Service': np.random.choice([1, -1], num_organized_cybercrime, p=[0.7, 0.3]), # High shortening service use
        'having_At_Symbol': np.random.choice([1, -1], num_organized_cybercrime, p=[0.2, 0.8]),
        'double_slash_redirecting': np.random.choice([1, -1], num_organized_cybercrime, p=[0.4, 0.6]),
        'Prefix_Suffix': np.random.choice([1, -1], num_organized_cybercrime, p=[0.3, 0.7]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_organized_cybercrime, p=[0.5, 0.3, 0.2]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_organized_cybercrime, p=[0.5, 0.4, 0.1]), # Low valid SSL
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_organized_cybercrime, p=[0.5, 0.3, 0.2]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_organized_cybercrime, p=[0.4, 0.4, 0.2]),
        'SFH': np.random.choice([-1, 0, 1], num_organized_cybercrime, p=[0.7, 0.2, 0.1]),
        'Abnormal_URL': np.random.choice([1, -1], num_organized_cybercrime, p=[0.7, 0.3]), # High abnormal URL
        'has_political_keyword': np.random.choice([1, -1], num_organized_cybercrime, p=[0.05, 0.95]) # Not typical for organized cybercrime
    }
    df_organized_cybercrime = pd.DataFrame(organized_cybercrime_data)
    df_organized_cybercrime['threat_actor_profile'] = 'Organized Cybercrime'

    # --- Hacktivist Profile (Opportunistic, political keywords, mixed tactics) ---
    hacktivist_data = {
        'having_IP_Address': np.random.choice([1, -1], num_hacktivist, p=[0.2, 0.8]),
        'URL_Length': np.random.choice([1, 0, -1], num_hacktivist, p=[0.3, 0.4, 0.3]),
        'Shortining_Service': np.random.choice([1, -1], num_hacktivist, p=[0.4, 0.6]),
        'having_At_Symbol': np.random.choice([1, -1], num_hacktivist, p=[0.3, 0.7]),
        'double_slash_redirecting': np.random.choice([1, -1], num_hacktivist, p=[0.2, 0.8]),
        'Prefix_Suffix': np.random.choice([1, -1], num_hacktivist, p=[0.4, 0.6]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_hacktivist, p=[0.4, 0.3, 0.3]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_hacktivist, p=[0.3, 0.4, 0.3]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_hacktivist, p=[0.4, 0.3, 0.3]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_hacktivist, p=[0.3, 0.4, 0.3]),
        'SFH': np.random.choice([-1, 0, 1], num_hacktivist, p=[0.5, 0.3, 0.2]),
        'Abnormal_URL': np.random.choice([1, -1], num_hacktivist, p=[0.3, 0.7]),
        'has_political_keyword': np.random.choice([1, -1], num_hacktivist, p=[0.8, 0.2]) # High political keyword use
    }
    df_hacktivist = pd.DataFrame(hacktivist_data)
    df_hacktivist['threat_actor_profile'] = 'Hacktivist'

    phishing_df = pd.concat([df_state_sponsored, df_organized_cybercrime, df_hacktivist], ignore_index=True)
    phishing_df['label'] = 1 # All these are malicious

    # --- Benign Data (similar to original, but with new feature) ---
    benign_data = {
        'having_IP_Address': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'URL_Length': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.6, 0.3]),
        'Shortining_Service': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_At_Symbol': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'double_slash_redirecting': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'Prefix_Suffix': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.4, 0.5]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_benign, p=[0.05, 0.15, 0.8]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'SFH': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.1, 0.8]),
        'Abnormal_URL': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'has_political_keyword': np.random.choice([1, -1], num_benign, p=[0.01, 0.99]) # Benign URLs usually don't have this
    }
    df_benign = pd.DataFrame(benign_data)
    df_benign['label'] = 0
    df_benign['threat_actor_profile'] = 'Benign' # Assign a profile for benign as well, though it won't be clustered

    final_df = pd.concat([phishing_df, df_benign], ignore_index=True)
    return final_df.sample(frac=1).reset_index(drop=True)

def train():
    # Paths for classification model
    model_path = 'models/phishing_url_detector'
    plot_path = 'models/feature_importance.png'

    # Paths for clustering model
    clustering_model_path = 'models/threat_actor_profiler'
    clustering_plot_path = 'models/cluster_plot.png'

    # Check if both models already exist to skip training
    if os.path.exists(model_path + '.pkl') and os.path.exists(clustering_model_path + '.pkl'):
        print("Models and plots already exist. Skipping training.")
        return

    data = generate_synthetic_data()
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/phishing_synthetic.csv', index=False)

    # --- Classification Model Training ---
    print("Initializing PyCaret Classification Setup...")
    # Target is 'label' (malicious/benign), ignore 'threat_actor_profile' for classification
    classification_s = classification_setup(data, target='label', session_id=42, verbose=False,
                                            ignore_features=['threat_actor_profile'])

    print("Comparing classification models...")
    best_classification_model = compare_models(n_select=1, include=['rf', 'et', 'lightgbm'])

    print("Finalizing classification model...")
    final_classification_model = finalize_model(best_classification_model)

    print("Saving classification feature importance plot...")
    os.makedirs('models', exist_ok=True)
    plot_classification_model(final_classification_model, plot='feature', save=True)
    # PyCaret saves it as 'Feature Importance.png', let's rename it
    if os.path.exists('Feature Importance.png'):
        os.rename('Feature Importance.png', plot_path)
    else:
        print("Warning: 'Feature Importance.png' not found after plotting.")

    print("Saving classification model...")
    save_model(final_classification_model, model_path)
    print(f"Classification model and plot saved successfully.")

    # --- Clustering Model Training ---
    print("\nInitializing PyCaret Clustering Setup...")
    # Filter for malicious data and keep 'threat_actor_profile' for later verification
    malicious_data = data[data['label'] == 1].copy()
    data_for_clustering = malicious_data.drop(columns=['label', 'threat_actor_profile'])

    # Using classification setup's preprocessed data to ensure consistent feature engineering if any
    clustering_s = clustering_setup(data_for_clustering, session_id=42, verbose=False, normalize=True)

    print("Creating clustering model (K-Means with 3 clusters)...")
    kmeans = create_model('kmeans', num_clusters=3)

    print("Assigning clusters to data...")
    # Changed assign_model to predict_model which takes a 'data' argument
    clustered_data_with_profiles = predict_clustering_model(kmeans, data=data_for_clustering)

    # Re-add the 'threat_actor_profile' column from the original malicious_data for verification
    # Ensure the indices align correctly
    clustered_data_with_profiles['threat_actor_profile'] = malicious_data['threat_actor_profile'].reset_index(drop=True)

    # Manual mapping verification (for development/debugging)
    # This helps understand which cluster ID corresponds to which synthetic profile
    print("\nCluster to Synthetic Profile Mapping (for verification):")
    print(clustered_data_with_profiles.groupby('Cluster')['threat_actor_profile'].value_counts())

    print("Saving clustering model...")
    save_clustering_model(kmeans, clustering_model_path)
    print(f"Clustering model saved successfully.")

    # Optional: Save a plot for clustering visualization (e.g., TSNE or PCA)
    # This can help visually confirm distinct clusters
    # plot_model(kmeans, plot='tsne', save=True)
    # if os.path.exists('t-SNE.png'):
    #    os.rename('t-SNE.png', clustering_plot_path)
    # else:
    #    print("Warning: 't-SNE.png' not found after plotting.")


if __name__ == "__main__":
    train()
