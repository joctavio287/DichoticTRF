from validation import main as validation_main_function
from figures import main as figures_main_function
from main import main as main_analysis_function
from load import main as load_main_function

if __name__ == "__main__":
    load_main_function()
    validation_main_function()
    main_analysis_function()
    figures_main_function()
