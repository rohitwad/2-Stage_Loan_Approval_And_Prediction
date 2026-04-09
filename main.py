# load functions
from app.loader import load_models
from app.predict import two_stage_predict
from app.utils import build_applicant_from_dict
import yaml

# Load models
configuration = yaml.safe_load(open('./config.yaml'))
cls, reg = load_models(configuration)

def run_cli():
    user_Input_Data = {
        'no_of_dependents': input("Enter number of dependents: "),
        'no_of_features': input("Enter number of features: "),
        'education': 'Graduate',
        'self_employed': 'No',
        'income_annum': 100000,
        'loan_amount': 30000,
        'loan_term': 20,
        'cibil_score': 150,
        'residential_assets_value': 300000,
        'commercial_assets_value': 200000,
        'luxury_assets_value': 0,
        'bank_asset_value': 55000
    }

    df = build_applicant_from_dict (user_Input_Data, list(cls.feature_names_in_))
    print(two_stage_predict(cls, reg, df))


if __name__ == "__main__":
    run_cli()
