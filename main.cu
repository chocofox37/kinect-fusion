#include <opencv2/opencv.hpp>

#include "level.cuh"

int main()
{
    cv::Mat matDepth = cv::imread("depth.png", cv::IMREAD_ANYDEPTH);
    matDepth.convertTo(matDepth, CV_32FC1);
    matDepth = matDepth / 5000.0;
    
    kf::Intrinsic intrinsic = {525.0, 525.0, 319.5, 239.5};
    kf::Level level(matDepth.cols, matDepth.rows, intrinsic);

    level.setDepthMap((kf::Depth*)matDepth.data);
    level.computeVertexMap();
    level.computeNormalMap();

    cv::Mat matVertex(matDepth.rows, matDepth.cols, CV_32FC3, level.vertexMap.data());
    cv::cvtColor(matVertex, matVertex, cv::COLOR_RGB2BGR);

    cv::Mat matNormal(matDepth.rows, matDepth.cols, CV_32FC3, level.normalMap.data());
    cv::cvtColor(matNormal, matNormal, cv::COLOR_RGB2BGR);

    cv::Mat matValidity(matDepth.rows, matDepth.cols, CV_8UC1, level.validityMask.data());

    cv::imshow("Depth Map", matDepth / 4.0);
    cv::imshow("Vertex Map", matVertex);
    cv::imshow("Normal Map", matNormal * 0.5 + 0.5);
    cv::imshow("Validity Mask", matValidity * 255);
    cv::waitKey(0);

    return 0;
}
