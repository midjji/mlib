#include <QVBoxLayout>

#include <opencv2/imgproc.hpp>

#include <mlib/gui/image_viewer_widget.h>


void TitledImageView::setLayoutVisible(bool visible){
    video->setVisible(visible);
    title->setVisible(visible);
}
void TitledImageView::set_image(cv::Mat1b im, std::string name){
    title->setText(name.c_str());
    //title->setText(("Time: "+toStr(frame->camera_driver_timestamp_ns) +" Frame Id: "+toStr(frame->camera_frame_id)).c_str());
    video->set_image(im);
}
void TitledImageView::set_image(cv::Mat3b bgr, std::string name){
    title->setText(name.c_str());
    video->set_image(bgr);
}
void SimpleImageView::set_image(cv::Mat1b im){
    QImage qim(im.data, im.cols, im.rows, (int)im.step, QImage::Format_Grayscale8);
    QPixmap pxmap = QPixmap::fromImage(qim);
    setPixmap(pxmap.scaled(size(), Qt::KeepAspectRatio));
    update();
    repaint();
}

SimpleImageView::SimpleImageView(QWidget* parent) : QLabel(parent)
{
    setAlignment(Qt::AlignHCenter);
}
void SimpleImageView::set_image(cv::Mat3b bgr)
{

    cv::Mat3b rgb;
    cv::cvtColor(bgr, rgb, cv::ColorConversionCodes::COLOR_BGR2RGB);

    QImage qim(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    QPixmap pxmap = QPixmap::fromImage(qim);
    setPixmap(pxmap.scaled(size(), Qt::KeepAspectRatio));
    update();
    repaint();
}

TitledImageView::TitledImageView(QWidget* parent): QWidget(parent){

    auto* layout = new QVBoxLayout(this);
    title = new QLabel(this);
    layout->addWidget(title,0);

    video = new SimpleImageView(this);
    layout->addWidget(video,1);

    title->setAlignment(Qt::AlignCenter);
    title->setText("Waiting");
}

