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

class Test
{
public:
	Test();

	Test(string);

	~Test();


	/*
	* getting the bag of words features through flann
	*/
	Mat getBOWFeatures(FlannBasedMatcher&, const Mat&, int);


	/*
	* reutrns the predicted class
	*/
	int getPredictedClass(const Mat& predictions);



	/*
	* Forming a confusion matrix
	*/
	vector<vector<int> > getConfusionMatrix(Ptr<ml::ANN_MLP>, const Mat&, const vector<int>&);


	/*
	* printing the confusion matrix
	*/
	void printConfusionMatrix(const vector<vector<int> >& confusionMatrix, const set<string>& classes);
	

	/**
	* Get the accuracy for a model (i.e., percentage of correctly predicted
	* test samples)
	*/
	float getAccuracy(const vector<vector<int> >&);


	/*
	* Reading test images
	*/
	void readImages(vec_iter, vec_iter);


	/*
	* copies all the files from a directory in a vector and returns it
	*/
	vector<string> getFilesInDirectory();


	/*
	* generates the class name i.e. cat or a dog
	* recieves filename to extract class from
	*/
	inline string getClassName(const string& );


	/*
	* extracts the features from an image
	* recieves the image Mat object
	*/
	Mat getDescriptors(const Mat&);



	/**
	* Transform a class name into an id
	*/
	int getClassId(const set<string>&, const string&);


	/*
	* THE BRAIN
	*/
	void flow(void);


private:
	string imagesDir;	//The path to the directory of our test images
	int networkInputSize;	//Using for determining the number of k-centroids
	Mat testSamples;	//Bag of word for each image
	vector<int> testOutputExpected;	//Expected output for each image
	set<string> classes;	//Names of classes present
	FlannBasedMatcher flann;	//Using flann to train it on the vocabulary and computing BOW because its much faster.
};

