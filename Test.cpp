#include "Test.h"


Test::Test()
{
}


Test::Test(string imageDir)
{
    this->imagesDir = imageDir;
    networkInputSize = 512;
}


Test::~Test()
{
}


/*
* Computing the bag of word for each image but this time through flann because
* its much faster.
*/
Mat Test::getBOWFeatures(FlannBasedMatcher& flann, const Mat& descriptors, int vocabularySize)
{
    Mat outputArray = Mat::zeros(Size(vocabularySize, 1), CV_32F);
    vector<DMatch> matches;
    flann.match(descriptors, matches);
    for (size_t j = 0; j < matches.size(); j++)
    {
        int visualWord = matches[j].trainIdx;
        outputArray.at<float>(visualWord)++;
    }
    return outputArray;
}


/*
* simply returns the the column index with the highest probability of it being the 
* expected output, which is already computed through predict method
*/
int Test::getPredictedClass(const Mat& predictions)
{
    float maxPrediction = predictions.at<float>(0);
    float maxPredictionIndex = 0;
    const float* ptrPredictions = predictions.ptr<float>(0);
    for (int i = 0; i < predictions.cols; i++)
    {
        float prediction = *ptrPredictions++;
        if (prediction > maxPrediction)
        {
            maxPrediction = prediction;
            maxPredictionIndex = i;
        }
    }
    return maxPredictionIndex;
}


/*
* Uses the mlp predict method to get the predicted outputs and then fills the
* confusion matrix accordingly
*/
vector<vector<int> > Test::getConfusionMatrix(Ptr<ml::ANN_MLP> mlp,
    const Mat& testSamples, const vector<int>& testOutputExpected)
{
    Mat testOutput;
    mlp->predict(testSamples, testOutput);
    vector<vector<int> > confusionMatrix(2, vector<int>(2));
    for (int i = 0; i < testOutput.rows; i++)
    {
        int predictedClass = getPredictedClass(testOutput.row(i));
        int expectedClass = testOutputExpected.at(i);
        confusionMatrix[expectedClass][predictedClass]++;
    }
    return confusionMatrix;
}


/*
* Simply prints the confusion matrix
*/
void Test::printConfusionMatrix(const vector<vector<int> >& confusionMatrix,
    const set<string>& classes)
{
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        cout << *it << " ";
    }
    cout << endl;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
        {
            cout << confusionMatrix[i][j] << " ";
        }
        cout << endl;
    }
}


/*
* uses the confusion matrix to get the accuracy of our predictions
*/
float Test::getAccuracy(const vector<vector<int> >& confusionMatrix)
{
    int hits = 0;
    int total = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        for (size_t j = 0; j < confusionMatrix.at(i).size(); j++)
        {
            if (i == j) hits += confusionMatrix.at(i).at(j);
            total += confusionMatrix.at(i).at(j);
        }
    }
    return hits / (float)total;
}


vector<string> Test::getFilesInDirectory()
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



inline string Test::getClassName(const string& filename)
{
    return filename.substr(filename.find_last_of('\\') + 1, 3);
}


Mat Test::getDescriptors(const Mat& img)
{
    Ptr<KAZE> kaze = KAZE::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    kaze->detectAndCompute(img, noArray(), keypoints, descriptors);
    return descriptors;
}


int Test::getClassId(const set<string>& classes, const string& classname)
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
* reads the images just like train method but here the bag of words is assigned
* to the respective image in this very part of the code
*/
void Test::readImages(vec_iter begin, vec_iter end)
{
    for (auto it = begin; it != end; ++it)
    {
        string filename = *it;
        cout << "Reading image " << filename << "..." << endl;
        Mat img = imread(filename, 0);
        if (img.empty())
        {
            cerr << "WARNING: Could not read image." << endl;
            continue;
        }
        string classname = getClassName(filename);
        cout << classname << endl;
        Mat descriptors = getDescriptors(img);
        // Get histogram of visual words using bag of words technique
        classes.insert(classname);
        Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
        normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, NORM_MINMAX, -1, Mat());
        testSamples.push_back(bowFeatures);
        testOutputExpected.push_back(getClassId(classes, classname));

    }
}


void Test::flow(void)
{
    cout << "Reading training set..." << endl;
    double start = (double)getTickCount();
    vector<string> files = getFilesInDirectory();
    random_shuffle(files.begin(), files.end());


    // Testing our neural network
    Ptr<ml::ANN_MLP> malp;
    // loading the trained model in malp
    malp = ml::ANN_MLP::load("mlp.yaml");
    // reading the vocabulary from file
    Mat vocab;
    FileStorage fs("vocabulary.yaml", FileStorage::READ);
    fs["vocabulary"] >> vocab;
    // Training flann according to our vocabulary for getting the visual words 
    cout << "Training FLANN..." << endl;
    start = getTickCount();
    flann.add(vocab);
    flann.train();
    cout << "Time elapsed in minutes: " << ((double)getTickCount() - start) / getTickFrequency() / 60.0 << endl;


    // Reading test set 
    cout << "Reading test set..." << endl;
    start = getTickCount();
    readImages(files.begin(), files.end());
    cout << "Time elapsed in minutes: " << ((double)getTickCount() - start) / getTickFrequency() / 60.0 << endl;

    
    // Get confusion matrix of the test set
    vector<vector<int> > confusionMatrix = getConfusionMatrix(malp, testSamples, testOutputExpected);

    
    // Get accuracy of our model
    cout << "Confusion matrix: " << endl;
    printConfusionMatrix(confusionMatrix, classes);
    cout << "Accuracy: " << getAccuracy(confusionMatrix) << endl;
}