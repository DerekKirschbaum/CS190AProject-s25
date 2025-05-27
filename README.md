# CS190AProject-s25

## Running the Code

To run the code:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DerekKirschbaum/CS190AProject-s25.git
   cd CS190AProject-s25

2. **Activate virtual environment:**
   ```bash
   source env311/bin/activate
   ```
   Create an env311 folder first if you don't have one already:
   ```bash
   python3.11 -m venv env311

3. **Install necessary packages:**
   ```bash
   pip install numpy torch torchvision facenet-pytorch matplotlib pillow insightface

4. **Run the program:**
   ```bash
   python FGSM_Perturbed_Images_Facenet.py

5. **Modify CONFIG:**\
    In the CONFIG section, modify variables to vary certain parameters influencing the model and our results
