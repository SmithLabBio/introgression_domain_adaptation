from secondary_contact import SecondaryContact, SecondaryContactData
from sim_wrapper.numpy_matrix_dataset import Dataset


d = Dataset(SecondaryContact, "secondaryContact1.json", 10)
print(d.__dict__)