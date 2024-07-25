import os
import pandas as pd

root_path = os.path.join(os.path.dirname(__file__), "../../../")

dataset = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:559]
triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722]
