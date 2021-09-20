#pragma once
#include <QLabel>
#include <opencv2/core/mat.hpp>

class SimpleImageView : public QLabel {
public:
    SimpleImageView(QWidget* parent = nullptr);
    void set_image(cv::Mat1b im);
    void set_image(cv::Mat3b bgr);
};

class TitledImageView : public QWidget {
public:
    TitledImageView(QWidget* parent = nullptr);
    void set_image(cv::Mat1b im, std::string name);
    void set_image(cv::Mat3b bgr, std::string name);
    void setLayoutVisible(bool visible);
    SimpleImageView* video;
    QLabel* title;

};
