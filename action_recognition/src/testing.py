from keras.applications.inception_v3 import InceptionV3

if __name__ == '__main__':
	base_model = InceptionV3(weights='imagenet', include_top=False)
	print(base_model.input.shape)