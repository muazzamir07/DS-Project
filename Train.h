#pragma once
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include"ImageData.h"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;
typedef vector<string>::const_iterator vec_iter;

class Train
{
public:
	Train();


	/*
	* This constructor recieves directory path and network input size
	*/
	Train(string, int);

	~Train();


	/*
	* copies all the files from a directory in a vector and returns it
	*/
	vector<string> getFilesInDirectory();

	
	/*
	* generates the class name i.e. cat or a dog
	* recieves filename to extract class from
	*/
	inline string getClassName(const string&);
	

	/*
	* extracts the features from an image
	* recieves the image Mat object
	*/
	Mat getDescriptors(const Mat&);


	/*
	* extracts the features from the training set and assigns a class name label to them
	*/
	void readImages(vec_iter, vec_iter);


	/**
	* Transform a class name into an id
	*/
	int getClassId(const set<string>&, const string&);


	/**
	* Get a binary code associated to a class
	*/
	Mat getClassCode( const string& );


	/**
	* Trains a neural network and returns it
	* recieves train samples and expected outputs for them
	*/
	Ptr<ml::ANN_MLP> getTrainedNeuralNetwork(const Mat&, const Mat&);



	/**
	* Save our obtained models (neural network, bag of words vocabulary
	* and class names) to use it later
	*/
	void saveModels(Ptr<ml::ANN_MLP>, const Mat&, const set<string>&);


	/*
	* This function pretty much is the brain of this class
	* Controls the flow of this class and trains the neural network
	*/
	void flow(void);

private:
	string imagesDir;	// The path of images
	int networkInputSize;	//Size of the input layer (which will also act as the no of visual words we want)
	Mat descriptorsSet;	//It will recieve the features extracted from all images
	vector<ImageData* > descriptorsMetadata;	//In this vector, each image data corresponding to DescriptorsSet will be stored
	set<string> classes;	//The classes which will be present. Cat | Dog
	Mat labels;		//Stores the K-means cluster indices
	Mat vocabulary;		//The vocabulary of our visual words
	Mat trainSamples;	//It will store the train samples i.e. histogram of visual words for each image
	Mat trainResponses;	//It will store the expected respose. Classic supervised learning...

};

