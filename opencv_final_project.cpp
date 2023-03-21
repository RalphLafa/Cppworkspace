
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;

#define PI 3.1415926
#define Width 480
#define Height 360

Point2f rotate_pixel(Mat& src, Point2f pos, double angle);
void image_rotate(Mat& src, Mat& dst, double angle);

void main() {
	Mat frame, flow, prevFrame, img;
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		printf("Couldn't open the web camera...\n");
		return;
	}

	CascadeClassifier cascade;
	cascade.load("C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");

	while (true) {
	
		Mat labels, stats, centroids;
		Mat motion;
		Mat origin;


		// Image Preprocessing
		capture >> frame;
		resize(frame, frame, Size(Width, Height));
		frame.copyTo(motion);
		resize(motion, img, Size(Width, Height));
		cvtColor(img, motion, CV_BGR2GRAY);
		Mat mask(Height, Width, CV_8UC1, Scalar(0));
		Mat dilated;
		mask.copyTo(dilated);
		Mat components;
		vector<Rect> faces;

		int x_start, y_start, x_end, y_end;

		Mat rot;
		frame.copyTo(rot);
		frame.copyTo(origin);

		cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50));

		if (!faces.size()) {
			for (double angle = -90; angle <= 90; angle += 30.) {
				image_rotate(frame, rot, angle);
				cascade.detectMultiScale(rot, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50));
				int minx = 1000, miny = 1000, maxx = 0, maxy = 0;
				if (faces.size()) {
					vector<Point2f> facepos, originpos;

					for (int i = 0; i < faces.size(); i++) {
					
						facepos.push_back(Point2f(faces[i].x, faces[i].y));
						facepos.push_back(Point2f(faces[i].x + faces[i].width, faces[i].y));
						facepos.push_back(Point2f(faces[i].x, faces[i].y + faces[i].height));
						facepos.push_back(Point2f(faces[i].x + faces[i].width, faces[i].y + faces[i].height));

				
						for (int i = 0; i < facepos.size(); i++) {
							originpos.push_back(rotate_pixel(rot, facepos[i], angle));
						}


						for (int i = 0; i < originpos.size(); i++) {
							minx = originpos[i].x > minx ? minx : originpos[i].x;
							miny = originpos[i].y > miny ? miny : originpos[i].y;
							maxx = originpos[i].x < maxx ? maxx : originpos[i].x;
							maxy = originpos[i].y < maxy ? maxy : originpos[i].y;

						}
					}


					if (minx > 0 && miny >= 0 && minx < frame.cols && miny < frame.rows) {
						Rect object(minx, miny, maxx - minx, maxy - miny);
						x_start = minx;
						y_start = miny;
						x_end = maxx;
						y_end = maxy;
						char str[20];
						sprintf(str, "face detected angle %d", int(angle));
						putText(frame, str, Point(minx, miny), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 0, 0), 2);
						rectangle(frame, object, Scalar(255, 0, 0), 2);
						rectangle(rot, Rect(faces[0].x, faces[0].y, faces[0].width, faces[0].height), Scalar(255, 0, 0), 2);
						break;
					}
				}
			}
		}
		else {	
			for (int i = 0; i < faces.size(); i++) {
				x_start = faces[i].x;
				y_start = faces[i].y;
				x_end = x_start + faces[i].width;
				y_end = y_start + faces[i].height;
				char str[20];
				sprintf(str, "face detected angle %d", 0);
				putText(frame, str, Point(faces[i].x, faces[i].y), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 0, 0), 2);
				rectangle(frame, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(0, 255, 0), 2);
			}
		}



		if (prevFrame.empty() == false) {
			// calculate optical flow
			calcOpticalFlowFarneback(prevFrame, motion, flow, 0.4, 1, 12, 2, 8, 1.2, 0);
			int x, y;

		
			for (y = y_start; y < y_end; y += 5) {
				for (x = x_start; x < x_end; x += 5) {
					const Point2f flowatxy = flow.at<Point2f>(y, x);

		
					Point tar = Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y));

					if (norm(tar - Point(x, y)) > 5) {
						line(frame, Point(x, y), tar, Scalar(255, 255, 127));
						line(mask, Point(x, y), tar, Scalar(255));
					}

				}
			}

			dilate(mask, dilated, Mat(), Point(-1, -1), 1);
			int num_labels = connectedComponentsWithStats(dilated, labels, stats, centroids);
			for (int i = 1; i < num_labels; i++) {
				int left = stats.at<int>(i, CC_STAT_LEFT);	// x pos
				int top = stats.at<int>(i, CC_STAT_TOP);	// y pos
				int width_label = stats.at<int>(i, CC_STAT_WIDTH);
				int height_label = stats.at<int>(i, CC_STAT_HEIGHT);

				rectangle(frame, Rect(left, top, width_label, height_label), Scalar(0, 0, 255));
				rectangle(mask, Rect(left, top, width_label, height_label), Scalar(255));
			}
			motion.copyTo(prevFrame);
		}
		else motion.copyTo(prevFrame);

		//imshow("origin", origin);
		//imshow("rot", rot);
		imshow("mask", mask);
		imshow("frame", frame);
		if (27 == cv::waitKey(5)) {
			frame.release();
			cv::destroyAllWindows();
		}

	}
}


void image_rotate(Mat& src, Mat& dst, double angle) {
	angle = angle * PI / 180.0;
	int y, x;
	for (y = 0; y < dst.rows; y++) {
		for (x = 0; x < dst.cols; x++) {
			double sinAngle = (double)(sin(angle));
			double cosAngle = (double)(cos(angle));
			int centerX = (int)(src.cols / 2);
			int centerY = (int)(src.rows / 2);

			// px = x * cos(angle) + y * sin(angle)
			double px = (double)(x - centerX) * (cosAngle)+(double)(y - centerY) * (sinAngle)+centerX;
			// py = -x * sin(angle) + y * cos(angle)
			double py = -(double)(x - centerX) * (sinAngle)+(double)(y - centerY) * (cosAngle)+centerY;

			int min_col = int(px);
			int min_row = int(py);
			int max_col = min_col + 1;
			int max_row = min_row + 1;

			double scale_mat[4];
			scale_mat[0] = px - (double)min_col;
			scale_mat[1] = 1 - scale_mat[0];
			scale_mat[2] = py - (double)min_row;
			scale_mat[3] = 1 - scale_mat[2];

			if (min_row > 0 && min_col > 0 && max_col < src.cols && max_row < src.rows) {
				Vec3b P1 = src.at<Vec3b>(min_row, min_col);
				Vec3b P2 = src.at<Vec3b>(min_row, max_col);
				Vec3b P3 = src.at<Vec3b>(max_row, min_col);
				Vec3b P4 = src.at<Vec3b>(max_row, max_col);

				double weight[4];
				weight[0] = scale_mat[1] * scale_mat[3];
				weight[1] = scale_mat[0] * scale_mat[3];
				weight[2] = scale_mat[1] * scale_mat[2];
				weight[3] = scale_mat[0] * scale_mat[2];

				dst.at<Vec3b>(y, x) = weight[0] * P1 + weight[1] * P2 + weight[2] * P3 + weight[3] * P4;
			}
			else
				dst.at<Vec3b>(y, x) = 0;
		}
	}
}

Point2f rotate_pixel(Mat& src, Point2f pos, double angle) {
	angle = angle * PI / 180.0;
	int y = pos.y;
	int x = pos.x;

	double sinAngle = (double)(sin(angle));
	double cosAngle = (double)(cos(angle));
	int centerX = (int)(src.cols / 2);
	int centerY = (int)(src.rows / 2);

	// px = x * cos(angle) + y * sin(angle)
	double px = (double)(x - centerX) * (cosAngle)+(double)(y - centerY) * (sinAngle)+centerX;
	// py = -x * sin(angle) + y * cos(angle)
	double py = -(double)(x - centerX) * (sinAngle)+(double)(y - centerY) * (cosAngle)+centerY;

	return Point2f(px, py);
}