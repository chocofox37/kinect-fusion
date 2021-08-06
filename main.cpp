#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

int main()
{
    std::cout << "Hello, world!" << std::endl;

    auto identity = Eigen::Matrix3d::Identity();
    std::cout << identity << std::endl;

    auto lena = cv::imread("lena.png");
    cv::imshow("Lena", lena);
    cv::waitKey(0);

    return 0;
}
