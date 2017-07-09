#include "stdafx.h"

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;

void detectAndDisplay(Mat frame, CascadeClassifier redCascade, CascadeClassifier greenCascade);

const String windowName = "Capture - TrafficLight detection";

const String pathToVideo = "D://YandexDisk//МИСиС//6семестр//Практика//red2green//positive//akn.025.026.left.avi";

const String redCascadeName = "C://ESD//HaarClassifier//x64//Debug//haarcascade_red_v11.xml";
const String greenCascadeName = "C://ESD//HaarClassifier//x64//Debug//haarcascade_green_v13.xml";

const double downscaleFactor = 1.1;

int main(void)
{
	VideoCapture capture = VideoCapture(pathToVideo);

	CascadeClassifier greenCascade;
	CascadeClassifier redCascade;

	if (!capture.isOpened()) { std::cout << "--(!)Error opening video capture\n"; return -1; }
	if (!redCascade.load(redCascadeName)) { std::cout << "--(!)Error loading red cascade\n"; return -1; };
	if (!greenCascade.load(greenCascadeName)) { std::cout << "--(!)Error loading green cascade\n"; return -1; };

	Mat frame;
	int counter = 0;
	while (capture.read(frame)) {
		if (frame.empty()) {
			std::cout << " --(!) No captured frame -- Break!";
			break;
		}

		if (counter % 30 == 0) {
			detectAndDisplay(frame, redCascade, greenCascade);
		}

		char c = (char)waitKey(10);
		if (c == 27) { break; } // Escape
		counter++;
	}
	return 0;
}

void detectAndDisplay(Mat frame, CascadeClassifier redCascade, CascadeClassifier greenCascade)
{
	Mat frameRed;
	Mat frameGreen;

	Mat bgr[3];
	split(frame, bgr);

	//		Green
	cvtColor(frame, frameGreen, COLOR_BGR2GRAY);
	frameGreen = frameGreen - 0.5 * bgr[2] + 0.2 * bgr[1];
	equalizeHist(frameGreen, frameGreen); // May be bad idea (especially for red)

	std::vector<Rect> greenTLights;
	greenCascade.detectMultiScale(frameGreen, greenTLights, downscaleFactor, 1, CV_HAAR_SCALE_IMAGE, Size(5, 10));
	for (size_t i = 0; i < greenTLights.size(); i++)
	{
		Point center(greenTLights[i].x + greenTLights[i].width / 2, greenTLights[i].y + greenTLights[i].height / 2);
		ellipse(frame, center, Size(greenTLights[i].width / 2, greenTLights[i].height / 2), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
		ellipse(frameGreen, center, Size(greenTLights[i].width / 2, greenTLights[i].height / 2), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
	}


	//		Red
	frameRed = bgr[2];
	equalizeHist(frameRed, frameRed);

	std::vector<Rect> redTLights;
	redCascade.detectMultiScale(frameRed, redTLights, downscaleFactor, 1, CV_HAAR_SCALE_IMAGE, Size(5, 10));
	for (size_t i = 0; i < redTLights.size(); i++)
	{
		Point center(redTLights[i].x + redTLights[i].width / 2, redTLights[i].y + redTLights[i].height / 2);
		ellipse(frame, center, Size(redTLights[i].width / 2, redTLights[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
		ellipse(frameRed, center, Size(redTLights[i].width / 2, redTLights[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
	}

	imshow(windowName, frame);
}