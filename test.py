from insightface.model_zoo import get_model

model = get_model('buffalo_l', download=True)
print(model)