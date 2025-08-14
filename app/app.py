# app.py
import streamlit as st
import pandas as pd
from pycaret.classification import load_model as load_classification_model, predict_model as predict_classification_model
from pycaret.clustering import load_model as load_clustering_model, predict_model as predict_clustering_model
from genai_prescriptions import generate_prescription
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="GenAI-Powered Phishing SOAR",
    page_icon="🛡️",
    layout="wide"
)
# --- Threat Actor Profiles ---
THREAT_ACTOR_PROFILES = {
    0: {
        "name": "State-Sponsored APT",
        "description": "Advanced Persistent Threat actors backed by nation-states",
        "characteristics": [
            "Highly sophisticated and well-resourced",
            "Uses valid SSL certificates to appear legitimate",
            "Employs subtle deception techniques like prefix/suffix manipulation",
            "Targets high-value government and corporate entities",
            "Maintains long-term persistence in networks"
        ],
        "typical_motivations": "Espionage, intellectual property theft, geopolitical advantage",
        "detection_difficulty": "High",
        "risk_color": "🔴"
    },
    1: {
        "name": "Organized Cybercrime",
        "description": "Profit-driven criminal organizations conducting mass-scale attacks",
        "characteristics": [
            "High-volume, automated attack campaigns",
            "Frequently uses URL shorteners and IP addresses",
            "Poor SSL certificate management",
            "Rapid domain cycling and infrastructure changes",
            "Mass-produced phishing kits and templates"
        ],
        "typical_motivations": "Financial gain, credential harvesting, ransomware deployment",
        "detection_difficulty": "Medium",
        "risk_color": "🟠"
    },
    2: {
        "name": "Hacktivist Group",
        "description": "Ideologically motivated actors conducting targeted campaigns",
        "characteristics": [
            "Opportunistic attack methods",
            "High use of political or social keywords",
            "Mixed sophistication levels",
            "Targets aligned with ideological beliefs",
            "Often announces attacks publicly"
        ],
        "typical_motivations": "Political activism, social justice, protest against organizations",
        "detection_difficulty": "Medium",
        "risk_color": "🟡"
    }
}
# --- Load Models and Feature Plot ---
@st.cache_resource
def load_assets():
    classification_model_path = 'models/phishing_url_detector'
    clustering_model_path = 'models/threat_actor_profiler'
    plot_path = 'models/feature_importance.png'

    classification_model = None
    clustering_model = None
    plot = None

    if os.path.exists(classification_model_path + '.pkl'):
        classification_model = load_classification_model(classification_model_path)
    if os.path.exists(clustering_model_path + '.pkl'):
        clustering_model = load_clustering_model(clustering_model_path)
    if os.path.exists(plot_path):
        plot = plot_path
    return classification_model, clustering_model, plot

classification_model, clustering_model, feature_plot = load_assets()

if not classification_model:
    st.error(
        "Classification model not found. Please wait for the initial training to complete, or check the container logs with `make logs` if the error persists."
    )
    st.stop()

if not clustering_model:
    st.warning(
        "Clustering model not found. Threat attribution will not be available. Please ensure `train_model.py` runs successfully."
    )

# --- Sidebar for Inputs ---
with st.sidebar:
    st.title("🔬 URL Feature Input")
    st.write("Describe the characteristics of a suspicious URL below.")

    # Using a dictionary to hold form values
    form_values = {
        'url_length': st.select_slider("URL Length", options=['Short', 'Normal', 'Long'], value='Long'),
        'ssl_state': st.select_slider("SSL Certificate Status", options=['Trusted', 'Suspicious', 'None'],
                                      value='Suspicious'),
        'sub_domain': st.select_slider("Sub-domain Complexity", options=['None', 'One', 'Many'], value='One'),
        'prefix_suffix': st.checkbox("URL has a Prefix/Suffix (e.g.,'-')", value=True),
        'has_ip': st.checkbox("URL uses an IP Address", value=False),
        'short_service': st.checkbox("Is it a shortened URL", value=False),
        'at_symbol': st.checkbox("URL contains '@' symbol", value=False),
        'abnormal_url': st.checkbox("Is it an abnormal URL", value=True),
        'has_political_keyword': st.checkbox("URL contains political keywords", value=False), # New input
    }

    st.divider()
    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI", "Grok"])
    submitted = st.button("💥 Analyze & Initiate Response", use_container_width=True, type="primary")

# --- Main Page ---
st.title("🛡️ GenAI-Powered SOAR for Phishing URL Analysis")
st.markdown("*From prediction to attribution - Advanced threat intelligence powered by AI*")

if not submitted:
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**How it works:**\n1. 📊 **Classify** - Determine if URL is malicious\n2. 🎯 **Attribute** - Identify likely threat actor profile\n3. 📋 **Prescribe** - Generate response plan")

    with col2:
        st.info(
            "**Threat Actor Profiles:**\n🔴 **State-Sponsored APT** - Nation-state actors\n🟠 **Organized Cybercrime** - Profit-driven groups\n🟡 **Hacktivist** - Ideologically motivated")

    st.info("Please provide the URL features in the sidebar and click 'Analyze' to begin.")
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(feature_plot,
                 caption="Feature importance from the trained RandomForest model. This shows which features the model weighs most heavily when making a prediction.")

else:
    # --- Data Preparation and Risk Scoring ---
    input_dict = {
        'having_IP_Address': 1 if form_values['has_ip'] else -1,
        'URL_Length': -1 if form_values['url_length'] == 'Short' else (
            0 if form_values['url_length'] == 'Normal' else 1),
        'Shortining_Service': 1 if form_values['short_service'] else -1,
        'having_At_Symbol': 1 if form_values['at_symbol'] else -1,
        'double_slash_redirecting': -1, # This feature is not exposed in UI, assumed constant
        'Prefix_Suffix': 1 if form_values['prefix_suffix'] else -1,
        'having_Sub_Domain': -1 if form_values['sub_domain'] == 'None' else (
            0 if form_values['sub_domain'] == 'One' else 1),
        'SSLfinal_State': -1 if form_values['ssl_state'] == 'None' else (
            0 if form_values['ssl_state'] == 'Suspicious' else 1),
        'Abnormal_URL': 1 if form_values['abnormal_url'] else -1,
        'URL_of_Anchor': 0, 'Links_in_tags': 0, 'SFH': 0, # These features are not exposed in UI, assumed constant
        'has_political_keyword': 1 if form_values['has_political_keyword'] else -1, # New feature
    }
    input_data_df = pd.DataFrame([input_dict])

    # Simple risk contribution for visualization
    risk_scores = {
        "Bad SSL": 25 if input_dict['SSLfinal_State'] < 1 else 0,
        "Abnormal URL": 20 if input_dict['Abnormal_URL'] == 1 else 0,
        "Prefix/Suffix": 15 if input_dict['Prefix_Suffix'] == 1 else 0,
        "Shortened URL": 15 if input_dict['Shortining_Service'] == 1 else 0,
        "Complex Sub-domain": 10 if input_dict['having_Sub_Domain'] == 1 else 0,
        "Long URL": 10 if input_dict['URL_Length'] == 1 else 0,
        "Uses IP Address": 5 if input_dict['having_IP_Address'] == 1 else 0,
        "Political Keyword": 10 if input_dict['has_political_keyword'] == 1 else 0,
    }
    risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Feature', 'Risk Contribution']).sort_values(
        'Risk Contribution', ascending=False)

    prescription = None
    threat_actor_profile = None

    # --- Analysis Workflow ---
    with st.status("Executing SOAR playbook...", expanded=True) as status:
        st.write("▶️ **Step 1: Predictive Analysis** - Running features through classification model.")
        time.sleep(1)
        # Predict malicious/benign using the classification model
        classification_prediction = predict_classification_model(classification_model, data=input_data_df)
        is_malicious = classification_prediction['prediction_label'].iloc[0] == 1

        verdict = "MALICIOUS" if is_malicious else "BENIGN"
        st.write(f"▶️ **Step 2: Verdict Interpretation** - Model predicts **{verdict}**.")
        time.sleep(1)

        if is_malicious:
            st.write(f"▶️ **Step 3: Prescriptive Analytics** - Engaging **{genai_provider}** for action plan.")
            try:
                prescription = generate_prescription(genai_provider, {k: v for k, v in input_dict.items()})
            except Exception as e:
                st.error(f"Failed to generate prescription: {e}")
                prescription = None
            time.sleep(1)

            if clustering_model:
                st.write("▶️ **Step 4: Threat Attribution** - Analyzing features with clustering model.")
                # Prepare data for clustering model (drop label and threat_actor_profile if present)
                # Ensure the column order and names match the training data for clustering
                clustering_input_features = input_data_df.drop(columns=[col for col in ['label', 'threat_actor_profile'] if col in input_data_df.columns])
                
                # IMPORTANT: The columns used for clustering must match those used during training
                # Reorder features for clustering prediction to match the training setup order
                # The order in train_model.py for clustering is determined by the DataFrame column order
                # Let's explicitly define it for consistency:
                clustering_features_order = [
                    'having_IP_Address', 'URL_Length', 'Shortining_Service',
                    'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
                    'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
                    'SFH', 'Abnormal_URL', 'has_political_keyword'
                ]
                
                # Select and reorder columns for prediction
                clustering_input_prepared = clustering_input_features[clustering_features_order]

                clustering_prediction = predict_clustering_model(clustering_model, data=clustering_input_prepared)
                predicted_cluster_id = clustering_prediction['Cluster'].iloc[0]

                # Map cluster ID to meaningful threat actor profile
                # This mapping might need to be adjusted based on actual clustering results from train_model.py
                # Based on the synthetic data generation logic:
                # Cluster 0 is likely Organized Cybercrime
                # Cluster 1 is likely State-Sponsored
                # Cluster 2 is likely Hacktivist
                cluster_to_profile_map = {
                    'Cluster 0': 'Organized Cybercrime',
                    'Cluster 1': 'State-Sponsored',
                    'Cluster 2': 'Hacktivist'
                }
                threat_actor_profile = cluster_to_profile_map.get(predicted_cluster_id, 'Unknown')
                st.write(f"▶️ **Step 5: Attribution Result** - Identified as **{threat_actor_profile}**.")
                time.sleep(1)
            else:
                st.write("⚠️ Skipping Threat Attribution: Clustering model not loaded.")
            status.update(label="✅ SOAR Playbook Executed Successfully!", state="complete", expanded=False)
        else:
            status.update(label="✅ Analysis Complete. No threat found.", state="complete", expanded=False)

    # --- Tabs for Organized Output ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 **Analysis Summary**", "📈 **Visual Insights**", "📜 **Prescriptive Plan**", "🕵️ **Threat Attribution**"])

    with tab1:
        st.subheader("Verdict and Key Findings")
        if is_malicious:
            st.error("**Prediction: Malicious Phishing URL**", icon="🚨")
        else:
            st.success("**Prediction: Benign URL**", icon="✅")

        st.metric("Malicious Confidence Score",
                  f"{classification_prediction['prediction_score'].iloc[0]:.2%}" if is_malicious else f"{1 - classification_prediction['prediction_score'].iloc[0]:.2%}")
        st.caption("This score represents the model's confidence in its prediction.")

    with tab2:
        st.subheader("Visual Analysis")
        st.write("#### Risk Contribution by Feature")
        st.bar_chart(risk_df.set_index('Feature'))
        st.caption("A simplified view of which input features contributed most to a higher risk score.")

        if feature_plot:
            st.write("#### Model Feature Importance (Global)")
            st.image(feature_plot,
                     caption="This plot shows which features the model found most important *overall* during its training.")

    with tab3:
        st.subheader("Actionable Response Plan")
        if prescription:
            st.success("A prescriptive response plan has been generated by the AI.", icon="🤖")
            st.json(prescription, expanded=False)  # Show the raw JSON for transparency

            st.write("#### Recommended Actions (for Security Analyst)")
            for i, action in enumerate(prescription.get("recommended_actions", []), 1):
                st.markdown(f"**{i}.** {action}")

            st.write("#### Communication Draft (for End-User/Reporter)")
            st.text_area("Draft", prescription.get("communication_draft", ""), height=150)
        else:
            st.info("No prescriptive plan was generated because the URL was classified as benign.")

    with tab4:
        st.subheader("Threat Actor Attribution")
        if is_malicious and threat_actor_profile:
            st.success(f"**Predicted Threat Actor: {threat_actor_profile}**", icon="👤")
            if threat_actor_profile == "State-Sponsored":
                st.markdown(
                    "These attacks are typically backed by nation-states, characterized by high sophistication, subtle deception, and long-term objectives like espionage or critical infrastructure disruption."
                )
            elif threat_actor_profile == "Organized Cybercrime":
                st.markdown(
                    "Driven by financial gain, these groups often use high-volume, noisy tactics like URL shortening and IP-based attacks, focusing on broad campaigns to maximize victims."
                )
            elif threat_actor_profile == "Hacktivist":
                st.markdown(
                    "Motivated by political or social causes, hacktivists employ opportunistic and sometimes crude methods, often incorporating keywords related to their agenda. Their goal is disruption and publicity."
                )
            else:
                st.info("No specific attribution description available for this profile.")
        elif not is_malicious:
            st.info("Threat actor attribution is only performed for URLs classified as **MALICIOUS**.")
        else:
            st.warning("Threat actor attribution could not be performed (e.g., clustering model not loaded or an error occurred).")
