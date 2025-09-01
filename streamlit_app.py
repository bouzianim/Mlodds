import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO

class ModelPredictor:
    """Load and use saved ML models for predictions"""
    
    def __init__(self, model_path='ml_models'):
        self.model_path = model_path
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all saved models and preprocessing objects"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model directory {self.model_path} not found")
        
        # Load models
        model_files = [f for f in os.listdir(self.model_path) 
                      if f.endswith('.pkl') and not f.startswith(('scaler_', 'encoder_'))]
        
        for file in model_files:
            model_name = file.replace('.pkl', '')
            self.models[model_name] = joblib.load(f"{self.model_path}/{file}")
        
        # Load scaler
        try:
            self.scaler = joblib.load(f"{self.model_path}/scaler_main.pkl")
        except:
            self.scaler = None
        
        # Load label encoders
        encoder_files = [f for f in os.listdir(self.model_path) if f.startswith('encoder_')]
        for file in encoder_files:
            encoder_name = file.replace('encoder_', '').replace('.pkl', '')
            self.label_encoders[encoder_name] = joblib.load(f"{self.model_path}/{file}")
    
    def engineer_features(self, h_odd, d_odd, a_odd, league='Premier League'):
        """Create feature vector for single match"""
        
        # Extract decimals
        h_decimal = int((h_odd % 1) * 100) if (h_odd % 1) > 0 else int(h_odd * 100) % 100
        d_decimal = int((d_odd % 1) * 100) if (d_odd % 1) > 0 else int(d_odd * 100) % 100
        a_decimal = int((a_odd % 1) * 100) if (a_odd % 1) > 0 else int(a_odd * 100) % 100
        
        sum_decimals = h_decimal + d_decimal + a_decimal
        
        # Calculate all features (must match training features exactly)
        features = {
            'h_odd': h_odd,
            'd_odd': d_odd,
            'a_odd': a_odd,
            'h_decimal': h_decimal,
            'd_decimal': d_decimal,
            'a_decimal': a_decimal,
            'sum_decimals': sum_decimals,
            'home_draw_ratio': h_odd / d_odd,
            'home_away_ratio': h_odd / a_odd,
            'draw_away_ratio': d_odd / a_odd,
            'draw_div4': d_decimal / 4,
            'sum_div10': sum_decimals / 10,
            'sum_div100': sum_decimals / 100,
        }
        
        # Pattern decimals
        features['draw_div4_decimal'] = int((features['draw_div4'] % 1) * 100) if features['draw_div4'] % 1 else int(features['draw_div4']) % 100
        features['sum_div10_decimal'] = int((features['sum_div10'] % 1) * 100) if features['sum_div10'] % 1 else int(features['sum_div10']) % 100
        features['sum_div100_decimal'] = int((features['sum_div100'] % 1) * 100) if features['sum_div100'] % 1 else int(features['sum_div100'] * 100) % 100
        
        # Pattern matches (binary)
        features['draw_div4_matches_h'] = 1 if str(int(features['draw_div4_decimal']))[0] == str(h_decimal)[0] else 0
        features['draw_div4_matches_d'] = 1 if str(int(features['draw_div4_decimal']))[0] == str(d_decimal)[0] else 0
        features['draw_div4_matches_a'] = 1 if str(int(features['draw_div4_decimal']))[0] == str(a_decimal)[0] else 0
        
        # Ratio signals
        features['ratio_away_signal'] = 1 if features['home_away_ratio'] > 1.5 else 0
        features['ratio_draw_signal'] = 1 if 0.8 <= features['home_draw_ratio'] <= 1.2 else 0
        features['ratio_home_signal'] = 1 if features['home_away_ratio'] < 0.7 else 0
        
        # League encoding
        if 'league' in self.label_encoders:
            try:
                features['league_encoded'] = self.label_encoders['league'].transform([league])[0]
            except:
                features['league_encoded'] = 0  # Default for unknown leagues
        else:
            features['league_encoded'] = 0
        
        # Statistical features
        odds_list = [h_odd, d_odd, a_odd]
        features['odds_variance'] = np.var(odds_list)
        features['odds_mean'] = np.mean(odds_list)
        features['favorite_odds'] = min(odds_list)
        
        return features
    
    def predict_single_match(self, h_odd, d_odd, a_odd, league='Premier League'):
        """Predict outcomes for a single match"""
        
        # Engineer features
        features_dict = self.engineer_features(h_odd, d_odd, a_odd, league)
        
        # Convert to DataFrame with correct column order
        feature_names = list(features_dict.keys())
        features_df = pd.DataFrame([list(features_dict.values())], columns=feature_names)
        
        # Scale features if scaler available
        if self.scaler:
            features_scaled = self.scaler.transform(features_df)
        else:
            features_scaled = features_df.values
        
        results = {}
        
        # Make predictions with all available models
        for model_name, model in self.models.items():
            target = 'FTR' if '_ftr' in model_name else 'HTR'
            algorithm = model_name.replace('_ftr', '').replace('_htr', '')
            
            # Choose appropriate input data
            if algorithm in ['logistic_regression', 'neural_network']:
                X_input = features_scaled
            else:
                X_input = features_df
            
            try:
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                confidence = max(probabilities)
                
                prob_dict = dict(zip(model.classes_, probabilities))
                
                if target not in results:
                    results[target] = {}
                
                results[target][algorithm] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': prob_dict
                }
                
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")
        
        return results, features_dict

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.models_loaded = False

def main():
    st.set_page_config(
        page_title="Football Match Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Football Match Prediction App")
    st.markdown("---")
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("🔧 Model Management")
        
        # Model path input
        model_path = st.text_input("Model Directory Path", value="ml_models")
        
        if st.button("Load Models", type="primary"):
            try:
                with st.spinner("Loading models..."):
                    st.session_state.predictor = ModelPredictor(model_path)
                    st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
                st.info(f"Loaded {len(st.session_state.predictor.models)} models")
                
                # Show loaded models
                st.subheader("Loaded Models:")
                for model_name in st.session_state.predictor.models.keys():
                    st.write(f"✅ {model_name}")
                    
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.session_state.models_loaded = False
    
    # Main content
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load the models first using the sidebar.")
        st.info("Make sure your ml_models folder contains the trained model files (.pkl)")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Match Prediction", "📊 Batch Predictions", "ℹ️ Model Info"])
    
    with tab1:
        st.header("Single Match Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Match Details")
            h_odd = st.number_input("Home Team Odds", min_value=1.0, max_value=50.0, value=2.0, step=0.01)
            d_odd = st.number_input("Draw Odds", min_value=1.0, max_value=50.0, value=3.5, step=0.01)
            a_odd = st.number_input("Away Team Odds", min_value=1.0, max_value=50.0, value=3.0, step=0.01)
            league = st.text_input("League", value="Premier League")
            
            if st.button("🔮 Make Prediction", type="primary"):
                try:
                    predictions, features = st.session_state.predictor.predict_single_match(
                        h_odd, d_odd, a_odd, league
                    )
                    st.session_state.current_predictions = predictions
                    st.session_state.current_features = features
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        
        with col2:
            if hasattr(st.session_state, 'current_predictions'):
                st.subheader("Predictions")
                
                predictions = st.session_state.current_predictions
                
                # Display FTR predictions
                if 'FTR' in predictions:
                    st.write("**Full Time Result (FTR):**")
                    ftr_df_data = []
                    for model_name, pred_data in predictions['FTR'].items():
                        ftr_df_data.append({
                            'Model': model_name,
                            'Prediction': pred_data['prediction'],
                            'Confidence': f"{pred_data['confidence']:.3f}",
                            'H': f"{pred_data['probabilities'].get('H', 0):.3f}",
                            'D': f"{pred_data['probabilities'].get('D', 0):.3f}",
                            'A': f"{pred_data['probabilities'].get('A', 0):.3f}"
                        })
                    
                    ftr_df = pd.DataFrame(ftr_df_data)
                    st.dataframe(ftr_df, use_container_width=True)
                
                # Display HTR predictions
                if 'HTR' in predictions:
                    st.write("**Half Time Result (HTR):**")
                    htr_df_data = []
                    for model_name, pred_data in predictions['HTR'].items():
                        htr_df_data.append({
                            'Model': model_name,
                            'Prediction': pred_data['prediction'],
                            'Confidence': f"{pred_data['confidence']:.3f}",
                            'H': f"{pred_data['probabilities'].get('H', 0):.3f}",
                            'D': f"{pred_data['probabilities'].get('D', 0):.3f}",
                            'A': f"{pred_data['probabilities'].get('A', 0):.3f}"
                        })
                    
                    htr_df = pd.DataFrame(htr_df_data)
                    st.dataframe(htr_df, use_container_width=True)
    
    with tab2:
        st.header("Batch Predictions from CSV")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="CSV should contain columns: B365H, B365D, B365A (and optionally FTR, HTR, League)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                
                required_cols = ['B365H', 'B365D', 'B365A']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    if st.button("🚀 Generate Predictions", type="primary"):
                        progress_bar = st.progress(0)
                        results_list = []
                        
                        for idx, row in df.iterrows():
                            h_odd, d_odd, a_odd = row['B365H'], row['B365D'], row['B365A']
                            league = row.get('League', 'Unknown')
                            
                            predictions, features = st.session_state.predictor.predict_single_match(
                                h_odd, d_odd, a_odd, league
                            )
                            
                            # Get best predictions
                            best_ftr = None
                            best_htr = None
                            
                            if 'FTR' in predictions:
                                best_ftr = max(predictions['FTR'].items(), key=lambda x: x[1]['confidence'])
                            if 'HTR' in predictions:
                                best_htr = max(predictions['HTR'].items(), key=lambda x: x[1]['confidence'])
                            
                            results_list.append({
                                'Match': idx + 1,
                                'Odds': f"{h_odd:.2f}/{d_odd:.2f}/{a_odd:.2f}",
                                'League': league,
                                'Best_FTR_Model': best_ftr[0] if best_ftr else 'N/A',
                                'Best_FTR_Prediction': best_ftr[1]['prediction'] if best_ftr else 'N/A',
                                'FTR_Confidence': f"{best_ftr[1]['confidence']:.3f}" if best_ftr else 'N/A',
                                'Best_HTR_Model': best_htr[0] if best_htr else 'N/A',
                                'Best_HTR_Prediction': best_htr[1]['prediction'] if best_htr else 'N/A',
                                'HTR_Confidence': f"{best_htr[1]['confidence']:.3f}" if best_htr else 'N/A',
                                'Actual_FTR': row.get('FTR', 'N/A'),
                                'Actual_HTR': row.get('HTR', 'N/A')
                            })
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results_list)
                        st.write("**Prediction Results:**")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tab3:
        st.header("Model Information")
        
        if st.session_state.predictor:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Loaded Models")
                for model_name, model in st.session_state.predictor.models.items():
                    with st.expander(f"🤖 {model_name}"):
                        st.write(f"**Type:** {type(model).__name__}")
                        if hasattr(model, 'classes_'):
                            st.write(f"**Classes:** {model.classes_}")
                        if hasattr(model, 'feature_importances_'):
                            st.write("**Has feature importances:** Yes")
                        else:
                            st.write("**Has feature importances:** No")
            
            with col2:
                st.subheader("🛠️ Preprocessing Objects")
                
                if st.session_state.predictor.scaler:
                    st.write("✅ **Scaler:** Loaded")
                else:
                    st.write("❌ **Scaler:** Not found")
                
                if st.session_state.predictor.label_encoders:
                    st.write("✅ **Label Encoders:** Loaded")
                    for encoder_name in st.session_state.predictor.label_encoders.keys():
                        st.write(f"   - {encoder_name}")
                else:
                    st.write("❌ **Label Encoders:** Not found")
                
                # Feature example
                if hasattr(st.session_state, 'current_features'):
                    st.subheader("🔍 Last Generated Features")
                    features_df = pd.DataFrame([st.session_state.current_features]).T
                    features_df.columns = ['Value']
                    st.dataframe(features_df)

if __name__ == "__main__":
    main()