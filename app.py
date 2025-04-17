import pickle
import gradio as gr
import numpy as np
import logging
from typing import Dict, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ForestFirePredictor:
    def __init__(self):
        try:
            with open("models/ridge.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("models/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Model and scaler loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def predict(
        self, Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region
    ) -> Dict[str, Union[str, float]]:
        try:
            validation = self._validate_inputs(
                Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region
            )
            if validation:
                return {"error": validation}

            input_data = np.array(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            )
            input_scaled = self.scaler.transform(input_data)
            prediction = float(self.model.predict(input_scaled)[0])
            level, color, recommendations = self._get_risk_assessment(prediction)

            return {
                "prediction": prediction,
                "risk_level": level,
                "color": color,
                "recommendations": recommendations,
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}

    def _validate_inputs(
        self, Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region
    ) -> str:
        if not (22 <= Temperature <= 42):
            return "Temperature must be between 22°C and 42°C."
        if not (21 <= RH <= 90):
            return "Relative Humidity must be between 21% and 90%."
        if not (6 <= Ws <= 29):
            return "Wind Speed must be between 6 and 29 km/h."
        if not (0 <= Rain <= 16.8):
            return "Rain must be between 0 and 16.8 mm."
        if not (28.6 <= FFMC <= 92.5):
            return "FFMC must be between 28.6 and 92.5."
        if not (1.1 <= DMC <= 65.9):
            return "DMC must be between 1.1 and 65.9."
        if not (0 <= ISI <= 18.5):
            return "ISI must be between 0 and 18.5."
        if Classes not in [0, 1]:
            return "Class must be 0 (No Fire) or 1 (Fire)."
        if Region not in [0, 1]:
            return "Region must be 0 (Bejaia) or 1 (Sidi-Bel Abbes)."
        return ""

    def _get_risk_assessment(self, fwi: float) -> Tuple[str, str, str]:
        if fwi <= 10:
            return (
                "Low Risk",
                "#d4edd8",  # Slightly deeper green for better contrast
                "• Regular monitoring recommended<br>• Standard fire prevention measures sufficient<br>• Good conditions for controlled burns if needed",
            )
        elif fwi <= 20:
            return (
                "Moderate Risk",
                "#fff0b3",  # Warmer yellow for better visibility
                "• Enhanced monitoring required<br>• Ensure fire breaks are maintained<br>• Review fire response procedures<br>• Avoid unnecessary burning activities",
            )
        else:
            return (
                "High Risk Level",
                "#ffd6d6",  # Slightly warmer red for better appeal
                "• Constant monitoring required<br>• All burning activities should be prohibited<br>• Emergency response teams should be on standby<br>• Public warning may be necessary<br>• Implement additional fire prevention measures",
            )


def create_interface():
    try:
        predictor = ForestFirePredictor()

        # Enhanced CSS with font matching for radio buttons and sliders
        custom_css = """
        /* Force light theme with custom styling */
        body, .gradio-container, .dark {
            color: #333 !important;
            background-color: #f8f9fa !important;
            font-family: 'Gill Sans', 'Optima', 'Segoe UI', Tahoma, sans-serif !important;
        }

        /* Base styling */
        #main-container {
            max-width: 850px;
            margin: 0 auto;
            padding: 25px;
        }

        h1 {
            color: #2b5876 !important;
            text-align: center;
            font-size: 32px;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
            margin-bottom: 25px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
            background: linear-gradient(135deg, #2b5876 0%, #4e8d7c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: sans-serif !important;
            font-style: normal;
        }

        /* Form container */
        .form-container {
            background-color: #ffffff !important;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            margin-bottom: 25px;
            border: 1px solid rgba(0,0,0,0.05);
        }

        /* Slider styling */
        .slider-container {
            margin-bottom: 22px;
        }

        /* Universal label styling for ALL form elements */
        .slider-container label, .radio-group label, .gr-form > div > label {
            display: block;
            font-weight: 700 !important; /* Matching the medium-bold style from the image */
            margin-bottom: 8px;
            color: #2b5876 !important;
            font-size: 15px;
            font-family: 'Gill Sans', 'Optima', sans-serif !important;
        }

        /* Specifically target the radio option text */
        .gr-form .gr-radio label span {
            font-weight: 700 !important;
            font-family: 'Gill Sans', 'Optima', sans-serif !important;
        }

        /* Fix for radio button option labels specifically */
        .gr-radio span, .gr-radio label {
            font-weight: 700 !important;
            font-family: 'Gill Sans', 'Optima', sans-serif !important;
        }

        /* Make sure the actual value texts match too */
        .gr-radio span:has(input[type="radio"]) {
            margin-right: 8px;
            font-weight: 700 !important;
        }

        /* Ensure info text styling matches */
        .slider-info, .gr-form .gr-input-label, .gr-form .gr-text-input + div {
            font-size: 13px;
            color: #6c757d !important;
            margin-top: 0;
            margin-bottom: 10px;
            font-style: italic;
            font-weight: normal !important;
        }

        /* Override Gradio's slider styles */
        input[type=range] {
            -webkit-appearance: none;
            width: 100%;
            height: 10px;
            border-radius: 6px;
            background: linear-gradient(to right, #e0f2f1, #b2dfdb);
            outline: none;
        }

        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2b5876 0%, #4e8d7c 100%);
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        input[type=range]::-moz-range-thumb {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2b5876 0%, #4e8d7c 100%);
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        /* Radio button styling */
        .radio-group {
            margin-bottom: 22px;
        }

        /* Target valid range text to make consistent */
        .gr-box > div:first-child + div, .gr-form .gr-form-text, .gr-radio + div {
            font-size: 13px !important;
            color: #6c757d !important;
            font-style: italic !important;
            font-weight: normal !important;
        }

        .radio-button {
            margin-right: 18px;
            margin-bottom: 12px;
            position: relative;
        }

        .radio-label {
            margin-left: 8px;
            font-size: 14px;
            font-weight: 700 !important;
        }

        /* Button styling */
        .predict-btn {
            background: linear-gradient(135deg, #2b5876 0%, #4e8d7c 100%) !important;
            color: white !important;
            padding: 14px;
            border: none;
            border-radius: 8px;
            width: 100%;
            font-size: 18px;
            font-weight: 700 !important;
            cursor: pointer;
            text-align: center;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            box-shadow: 0 4px 10px rgba(43, 88, 118, 0.3);
            transition: all 0.3s ease;
        }

        .predict-btn:hover {
            box-shadow: 0 6px 14px rgba(43, 88, 118, 0.4) !important;
            transform: translateY(-2px);
        }

        /* Result container */
        .result-container {
            padding: 25px;
            border-radius: 12px;
            margin-top: 25px;
            text-align: center;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
        }

        /* Info box */
        .info-box {
            background: linear-gradient(135deg, #e0f7fa 0%, #e0f2f1 100%) !important;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            color: #333 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.05);
        }

        .info-box h3 {
            color: #2b5876 !important;
            margin-top: 0;
            font-size: 20px;
            font-weight: 700 !important;
        }

        .info-box p {
            font-size: 15px;
            line-height: 1.6;
        }

        .risk-categories {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .risk-category {
            padding: 10px 18px;
            border-radius: 8px;
            text-align: center;
            font-weight: 700 !important;
            color: #333 !important;
            font-size: 14px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            margin: 5px;
            flex-grow: 1;
            border: 2px solid rgba(255,255,255,0.8);
        }

        .risk-category.low {
            background: linear-gradient(135deg, #d4edd8 0%, #c8e6c9 100%) !important;
        }

        .risk-category.moderate {
            background: linear-gradient(135deg, #fff8e1 0%, #fff0b3 100%) !important;
        }

        .risk-category.high {
            background: linear-gradient(135deg, #ffebee 0%, #ffd6d6 100%) !important;
        }

        /* Risk level styling */
        .risk-level {
            font-size: 22px;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            margin-bottom: 18px;
            text-transform: uppercase;
        }

        .risk-level.high {
            color: #c62828 !important;
            text-shadow: 1px 1px 3px rgba(198, 40, 40, 0.2);
        }

        .risk-level.moderate {
            color: #ef6c00 !important;
            text-shadow: 1px 1px 3px rgba(239, 108, 0, 0.2);
        }

        .risk-level.low {
            color: #2e7d32 !important;
            text-shadow: 1px 1px 3px rgba(46, 125, 50, 0.2);
        }

        /* Result box text improvements */
        .result-fwi {
            font-size: 18px;
            font-weight: 700 !important;
            margin-bottom: 12px;
            color: #2b5876 !important;
        }
        
        .result-description {
            font-style: italic;
            margin-bottom: 15px;
            font-size: 15px;
            font-weight: 700 !important;
        }
        
        .recommendations-title {
            font-weight: 700 !important;
            margin-top: 18px;
            margin-bottom: 10px;
            font-size: 16px;
            color: #2b5876 !important;
        }
        
        .recommendations-list {
            text-align: left;
            list-style-type: none;
            padding-left: 10px;
        }
        
        .recommendations-list li {
            position: relative;
            padding-left: 22px;
            margin-bottom: 8px;
            line-height: 1.5;
            font-weight: 700 !important;
        }
        
        .recommendations-list li:before {
            content: '•';
            position: absolute;
            left: 0;
            color: #2b5876 !important;
            font-weight: bold;
        }
        
        /* Force text legibility */
        * {
            color: #333 !important;
        }
        
        /* Custom color overrides for specific elements */
        .predict-btn * {
            color: white !important;
        }

        /* Override any Gradio-specific styles for radio buttons to match slider labels */
        .gr-radio-row label, .gr-radio label {
            font-weight: 700 !important;
            color: #2b5876 !important;
        }
        
        /* Target the specific radio labels even more specifically */
        .gr-form .gr-radio label span[data-testid] {
            font-weight: 700 !important;
            font-family: 'Gill Sans', 'Optima', sans-serif !important;
            color: #333 !important;
        }
        
        /* Match the exact font weight for the "Valid range" text */
        .gr-form .gr-panel-content div:has(+ input[type="range"]) + div {
            font-size: 13px !important;
            color: #6c757d !important;
            font-style: italic !important;
            font-weight: normal !important;
        }

        /* Radio button styling */
        .button-radio input[type="radio"]:checked + label {
            background-color: #fd7e14;
            color: white;
            border-color: #fd7e14;
        }
        
        .button-radio label {
            background-color: white;
            color: black;
            padding: 12px 24px;
            border: 2px solid #ced4da;
            border-radius: 8px;
            margin: 5px;
            cursor: pointer;
            display: inline-block;
            min-width: 100px;
            text-align: center;
            transition: all 0.3s ease;
            font-weight: 800 !important;  /* Extra bold */
        }
        
        .button-radio label:hover {
            background-color: #fff3e6;
        }

        /* Make labels extra bold */
        .gr-form > div > label {
            font-weight: 800 !important;  /* Extra bold */
            font-size: 1.1em !important;  /* Slightly larger */
        }

        /* Make info text extra bold */
        .gr-form .gr-input-label, 
        .gr-form .gr-text-input + div,
        .gr-form .gr-slider + div,
        .gr-form .gr-radio-row + div {
            font-weight: 800 !important;  /* Extra bold */
        }

        /* Make all input values extra bold */
        .gr-box > div,
        .gr-form input[type="number"],
        .gr-form .gr-text-input,
        .gr-slider-value {
            font-weight: 800 !important;  /* Extra bold */
        }

        /* Specifically target radio labels and descriptions */
        .gr-radio-row label,
        .gr-form .gr-radio-row + div,
        .gr-form .gr-radio-row label span {
            font-weight: 800 !important;  /* Extra bold */
            color: #000000 !important;    /* Darker text */
        }

        /* Make slider labels and values extra bold */
        .gr-slider-label, 
        .gr-slider-value {
            font-weight: 800 !important;  /* Extra bold */
        }
        """

        def on_predict(temp, rh, ws, rain, ffmc, dmc, isi, classes, region):
            result = predictor.predict(
                temp, rh, ws, rain, ffmc, dmc, isi, classes, region
            )

            if "error" in result:
                return gr.update(visible=True, value=f"⚠️ {result['error']}"), gr.update(
                    visible=False
                )

            risk_class = (
                "low"
                if result["prediction"] <= 10
                else "moderate" if result["prediction"] <= 20 else "high"
            )

            # Enhanced result HTML with better visual hierarchy and styling
            html = f"""
            <div style="background: linear-gradient(135deg, {result['color']} 0%, {'#c8e6c9' if risk_class == 'low' else '#ffe082' if risk_class == 'moderate' else '#ffcdd2'} 100%); 
                       padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 6px 18px rgba(0,0,0,0.08); border: 2px solid rgba(255,255,255,0.8);">
                <div class="result-fwi" style="font-size: 22px; font-weight: 800 !important; color: #2b5876 !important;">
                    Predicted Fire Weather Index: {result['prediction']:.2f}
                </div>
                
                <div class="risk-level {risk_class}" style="font-size: 26px; font-weight: 800 !important; letter-spacing: 1px; margin: 18px 0; 
                     color: {'#c62828' if risk_class == 'high' else '#ef6c00' if risk_class == 'moderate' else '#2e7d32'} !important; 
                     text-shadow: 1px 1px 3px rgba(0,0,0,0.1); text-transform: uppercase;">
                    {result['risk_level']}
                </div>
                
                {f'<div class="result-description" style="font-style: italic; font-size: 16px; margin-bottom: 20px; color: #c62828 !important; font-weight: 800 !important;">⚠️ Dangerous fire conditions present. Immediate precautions necessary.</div>' if risk_class == "high" else ""}
                
                <div class="recommendations-title" style="font-weight: 800 !important; margin-top: 20px; font-size: 18px; color: #2b5876 !important;">
                    Recommended Actions:
                </div>
                
                <ul class="recommendations-list" style="text-align: left; list-style-type: none; padding-left: 10px;">
                    {result['recommendations'].replace("• ", "<li style='position: relative; padding-left: 25px; margin-bottom: 10px; line-height: 1.6; font-size: 15px; font-weight: 800 !important;'><span style='position: absolute; left: 0; color: #2b5876 !important; font-weight: bold;'>•</span>").replace("<br>", "</li>")}
                </ul>
            </div>
            """
            return gr.update(visible=False), gr.update(visible=True, value=html)

        # Create an enhanced intro section
        intro_html = """
        <div style="background: linear-gradient(135deg, #e0f7fa 0%, #e0f2f1 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px; 
             box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);">
            <h3 style="color: #2b5876 !important; font-size: 22px; font-weight: 800 !important; margin-top: 0; margin-bottom: 15px;">About the Fire Weather Index (FWI)</h3>
            
            <p style="color: #333 !important; font-size: 15px; line-height: 1.6; margin-bottom: 20px; font-weight: 800 !important;">
                The Fire Weather Index (FWI) is a numerical rating that combines various weather conditions to estimate fire 
                potential. It considers temperature, humidity, wind, rain, and other environmental factors to assess fire risk in forest
                areas. Use this tool to calculate the current fire risk in your region based on weather conditions.
            </p>
            
            <div style="display: flex; justify-content: space-between; margin-top: 20px; flex-wrap: wrap;">
                <div style="background: linear-gradient(135deg, #d4edd8 0%, #c8e6c9 100%); padding: 12px 18px; border-radius: 8px; 
                     text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin: 5px; flex-grow: 1; border: 2px solid rgba(255,255,255,0.8);">
                    <strong style="color: #2e7d32 !important; font-size: 16px; font-weight: 800 !important;">Low Risk (0-10)</strong>
                </div>
                <div style="background: linear-gradient(135deg, #fff8e1 0%, #fff0b3 100%); padding: 12px 18px; border-radius: 8px; 
                     text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin: 5px; flex-grow: 1; border: 2px solid rgba(255,255,255,0.8);">
                    <strong style="color: #ef6c00 !important; font-size: 16px; font-weight: 800 !important;">Moderate Risk (11-20)</strong>
                </div>
                <div style="background: linear-gradient(135deg, #ffebee 0%, #ffd6d6 100%); padding: 12px 18px; border-radius: 8px; 
                     text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin: 5px; flex-grow: 1; border: 2px solid rgba(255,255,255,0.8);">
                    <strong style="color: #c62828 !important; font-size: 16px; font-weight: 800 !important;">High Risk (>20)</strong>
                </div>
            </div>
        </div>
        """

        # Use gr.Blocks with explicit theme="light"
        with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
            gr.HTML(
                "<h1>Forest Fire Weather Index Prediction</h1>",
                elem_id="main-container",
            )

            # Add enhanced intro section
            gr.HTML(intro_html)

            # Create the main form container
            with gr.Column(elem_classes=["form-container"]):
                # Switch back to sliders instead of number inputs
                temp = gr.Slider(
                    label="Temperature (°C)",
                    minimum=22,
                    maximum=42,
                    step=0.1,
                    value=32,  # Midpoint: (22 + 42) / 2 = 32°C
                    info="Valid range: 22°C to 42°C",
                    elem_classes=["slider-container"],
                )
                rh = gr.Slider(
                    label="Relative Humidity (RH)",
                    minimum=21,
                    maximum=90,
                    step=0.1,
                    value=55.5,  # Midpoint: (21 + 90) / 2 = 55.5%
                    info="Valid range: 21% to 90%",
                    elem_classes=["slider-container"],
                )
                ws = gr.Slider(
                    label="Wind Speed",
                    minimum=6,
                    maximum=29,
                    step=0.1,
                    value=17.5,  # Midpoint: (6 + 29) / 2 = 17.5 km/h
                    info="Valid range: 6 to 29 km/h",
                    elem_classes=["slider-container"],
                )
                rain = gr.Slider(
                    label="Rain",
                    minimum=0,
                    maximum=16.8,
                    step=0.1,
                    value=8.4,  # Midpoint: (0 + 16.8) / 2 = 8.4 mm
                    info="Valid range: 0 to 16.8 mm",
                    elem_classes=["slider-container"],
                )
                ffmc = gr.Slider(
                    label="Fine Fuel Moisture Code (FFMC)",
                    minimum=28.6,
                    maximum=92.5,
                    step=0.1,
                    value=60.55,  # Midpoint: (28.6 + 92.5) / 2 = 60.55
                    info="Valid range: 28.6 to 92.5",
                    elem_classes=["slider-container"],
                )
                dmc = gr.Slider(
                    label="Duff Moisture Code (DMC)",
                    minimum=1.1,
                    maximum=65.9,
                    step=0.1,
                    value=33.5,  # Midpoint: (1.1 + 65.9) / 2 = 33.5
                    info="Valid range: 1.1 to 65.9",
                    elem_classes=["slider-container"],
                )
                isi = gr.Slider(
                    label="Initial Spread Index (ISI)",
                    minimum=0,
                    maximum=18.5,
                    step=0.1,
                    value=9.25,  # Midpoint: (0 + 18.5) / 2 = 9.25
                    info="Valid range: 0 to 18.5",
                    elem_classes=["slider-container"],
                )

                # Use radio buttons for fire class and region
                classes = gr.Radio(
                    choices=[0, 1],
                    value=0,  # Binary choice - keeping at 0 (No Fire)
                    label="Fire Class",
                    info="0: No Fire, 1: Fire",
                    elem_classes=["radio-group"],
                )
                region = gr.Radio(
                    choices=[0, 1],
                    value=0,  # Binary choice - keeping at 0 (Bejaia)
                    label="Region",
                    info="0: Bejaia, 1: Sidi-Bel Abbes",
                    elem_classes=["radio-group"],
                )

                predict_btn = gr.Button(
                    "Predict Fire Weather Index", elem_classes=["predict-btn"]
                )

                error_msg = gr.Markdown(visible=False)
                result_html = gr.HTML(visible=False)

                # Connect the prediction function
                predict_btn.click(
                    fn=on_predict,
                    inputs=[temp, rh, ws, rain, ffmc, dmc, isi, classes, region],
                    outputs=[error_msg, result_html],
                )

        return demo

    except Exception as e:
        logger.error(f"Application startup error: {e}")
        raise


if __name__ == "__main__":
    try:
        demo = create_interface()
        demo.launch(share=False)  
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
