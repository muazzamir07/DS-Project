#include "Train.h"
#include"ImageData.h"

Train::Train()
{
}


Train::Train(string imagesDir, int networkInputSize)
{
    this->imagesDir = imagesDir;
    this->networkInputSize = networkInputSize;
}


Train::~Train()
{

}


/*
* Iterates over each file in the directory and pushes it in a vector(files)
*/
vector<string> Train::getFilesInDirectory()
{
    vector<string> files;
    fs::path root(imagesDir);
    fs::directory_iterator it_end;
    for (fs::directory_iterator it(root); it != it_end; ++it)
    {
        if (fs::is_regular_file(it->path()))
        {
            files.push_back(it->path().string());
        }
    }
    return files;
}



/*
* detects a "\" character and then returns the next three letters (cat or dog)
* This will be the classname of this image which will help in supervised learning
*/
inline string Train::getClassName(const string& filename)
{
    return filename.substr(filename.find_last_of('\\') + 1, 3);
}


/*
* Extracts the keypoint features of the images through kaze
*/
Mat Train::getDescriptors(const Mat& img)
{
    Ptr<KAZE> kaze = KAZE::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    kaze->detectAndCompute(img, noArray(), keypoints, descriptors);
    return descriptors;
}



/*
* Iterates over each image, extracts its classname and features and adds it in
* the DescriptorsSet. Then creates an ImageData object and assigns to it, its classname
* and then assigns a matrix of zeros (BOW Features not extracted yet) to
* bowFeatures. It fills descriptors metadata in a way that each row of DescriptorsSet corresponds
* to its classname and visual words...
*/
void Train::readImages(vec_iter begin, vec_iter end)
{
    int iteration = 0;
    for (auto it = begin; it != end; ++it)
    {
        string filename = *it;
        cout << iteration<<"  Reading image " << filename << "..." << endl;
        Mat img = imread(filename, 0);
        if (img.empty())
        {
            cerr << "WARNING: Could not read image." << endl;
            continue;
        }
        string classname = getClassName(filename);
        cout << classname << endl;
        Mat descriptors = getDescriptors(img);
        classes.insert(classname);
        // Append to the list of descriptors
        descriptorsSet.push_back(descriptors);
        // Append metadata to each extracted feature
        ImageData* data = new ImageData();
        data->classname = classname;
        data->bowFeatures = Mat::zeros(Size(networkInputSize, 1), CV_32F);
        for (int j = 0; j < descriptors.rows; j++)
        {
            descriptorsMetadata.push_back(data);
        }
        iteration++;
    }
}


/*
* Generates an ID for each class.
*/
int Train::getClassId(const set<string>& classes, const string& classname)
{
    int index = 0;
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        if (*it == classname) break;
        ++index;
    }
    return index;
}


/*
* Generates the binary code for each class
* for e.g. 10 for cat, 01 for  dog
*/
Mat Train::getClassCode(const string& classname)
{
    Mat code = Mat::zeros(Size((int)classes.size(), 1), CV_32F);
    int index = getClassId(classes, classname);
    code.at<float>(index) = 1;
    return code;
}


/*
* Trains our neural network. First creates an object of ANN_MLP and then sets the 
* layer sizes. In this neural network, there will be:
* 1 input layers
* 2 hidden layers
* and 1 output layer
* The no of neurons in input layer for each image will be same as the number of visual words
* of the image. (512 in our case)
* Sigmoid activation function is used here.
* And then simply the network is trained according to the responses
*/
Ptr<ml::ANN_MLP> Train::getTrainedNeuralNetwork(const Mat& trainSamples, const Mat& trainResponses)
{
    int networkInputSize = trainSamples.cols;
    int networkOutputSize = trainResponses.cols;
    Ptr<ml::ANN_MLP> mlp = ml::ANN_MLP::create();
    vector<int> layerSizes = { networkInputSize, networkInputSize / 2, networkInputSize/4, networkOutputSize };
    mlp->setLayerSizes(layerSizes);
    mlp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    mlp->train(trainSamples, ml::ROW_SAMPLE, trainResponses);
    return mlp;
}


/*
* Saves the model and vocabulary so that the test set can use it
*/
void Train::saveModels(Ptr<ml::ANN_MLP> mlp, const Mat& vocabulary, const set<string>& classes)
{
    mlp->save("malp.yaml");
    FileStorage fs("vocabulary.yaml", FileStorage::WRITE);
    fs << "vocabulary" << vocabulary;
    fs.release();
    ofstream classesOutput("classes.txt");
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        classesOutput << getClassId(classes, *it) << "\t" << *it << endl;
    }
    classesOutput.close();
}


/*
* The brain of Train Class. It controls the flow which our images will be processed 
* and neural network will be trained.
*/
void Train::flow()
{
    cout << "Reading training set..." << endl;
    double start = (double)getTickCount();
    vector<string> files = getFilesInDirectory();
    //shuffling to prevent bias:
    random_shuffle(files.begin(), files.end()); 
    

    // assigning each image its features and classnames and mantaining a descritorsSet 
    // for our vocabulary
    readImages(files.begin(), files.end());
    cout << "Time elapsed in minutes: " << ((double)getTickCount() - start) / getTickFrequency() / 60.0 << endl;


    // Now creating the vocabulary (all possible visual words)
    cout << "Creating vocabulary..." << endl;
    start = (double)getTickCount();
    // Using k-means to find k centroids (the words of our vocabulary)
    // This functions find k-centroids for each image and stops when the max iteration is reached
    kmeans(descriptorsSet, networkInputSize, labels, TermCriteria(TermCriteria::EPS +
    TermCriteria::MAX_ITER, 10, 0.01), 1, KMEANS_PP_CENTERS, vocabulary);
    // No need to keep it on memory anymore
    descriptorsSet.release();
    cout << "Time elapsed in minutes: " << ((double)getTickCount() - start) / getTickFrequency() / 60.0 << endl;

    
    // Now creating a histogram (bag of words) for for each image
    cout << "Getting histograms of visual words..." << endl;
    int* ptrLabels = (int*)(labels.data);
    int size = labels.rows * labels.cols;
    for (int i = 0; i < size; i++)
    {
        int label = *ptrLabels++;
        ImageData* data = descriptorsMetadata[i];
        data->bowFeatures.at<float>(label)++;

    }


    // Filling matrices to be used by the neural network
    cout << "Preparing neural network..." << endl;
    
    set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
    for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); )
    {
        ImageData* data = *it;
        Mat normalizedHist;
        normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, NORM_MINMAX, -1, Mat());
        trainSamples.push_back(normalizedHist);
        trainResponses.push_back(getClassCode( data->classname));
        delete* it; // clear memory
        it++;
    }
    descriptorsMetadata.clear();


     // Training neural network
    cout << "Training neural network..." << endl;
    start = getTickCount();
    Ptr<ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
    cout << "Time elapsed in minutes: " << ((double)getTickCount() - start) / getTickFrequency() / 60.0 << endl;

    
    // We can clear memory now 
    trainSamples.release();
    trainResponses.release();


    // Save models
    cout << "Saving models..." << endl;
    saveModels(mlp, vocabulary, classes);
}