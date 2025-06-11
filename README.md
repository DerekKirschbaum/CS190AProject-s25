# CS190AProject-s25

This repo contains all of the code and figures that we used in all of the experiments, as well as our final paper, "[Exploring the Impact of Model Architecture on Adversarial Transferability in Facial Recognition](https://github.com/DerekKirschbaum/CS190AProject-s25/blob/main/Adversarial_Machine_Learning_Report%20(9).pdf)".

## Running the Code

To run the code:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DerekKirschbaum/CS190AProject-s25.git
   cd CS190AProject-s25

2. **Activate virtual environment:**
   
   Create an env311 folder first if you don't have one already:
   ```bash
   python3.11 -m venv env311
   ```

   Then run:
   ```bash
   source env311/bin/activate
   ```

4. **Install necessary packages:**
   ```bash
   pip install numpy torch torchvision facenet-pytorch matplotlib pillow insightface transforms 

5. **Run the program:**
   To test specific types of attacks, modify epsilon values, or modify the models being tested, edit perturbation_testing.py and then run:
   ```bash
   python perturbation_testing.py
   ```
   To view visuals of specific gradients or to save perturbed images, edit visual.py and then run:
   ```bash
   python visual.py
   ```
   If you would like to vary the dataset and establish a new random seed for train/validation/test split, edit preprocess_data.py and then run:
   ```bash
   python preprocess_data.py
   python build.py
   ```

