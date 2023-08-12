#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <kht/kht.hpp>
#include <random>
#include <vector>

std::double_t const DEGREES_TO_RADIANS = std::atan(1.0) / 45.0;

// Function to find intersections of lines
std::vector<cv::Point2f> findIntersections(const kht::ListOfLines& lines, std::int32_t width, std::int32_t height, std::int32_t relevant_lines) 
{
    std::vector<cv::Point2f> intersections;
    for (std::size_t i = 0; i < relevant_lines; ++i)
    {
        for (std::size_t j = i + 1; j < relevant_lines; ++j) 
        {
            auto const& line1 = lines[i];
            auto const& line2 = lines[j];

            std::double_t denom = sin(line1.theta*DEGREES_TO_RADIANS)*cos(line2.theta*DEGREES_TO_RADIANS) - cos(line1.theta*DEGREES_TO_RADIANS)*sin(line2.theta*DEGREES_TO_RADIANS);
            if (std::abs(denom) < 0.95) continue; 

            std::double_t x = (line2.rho*sin(line1.theta*DEGREES_TO_RADIANS) - line1.rho*sin(line2.theta*DEGREES_TO_RADIANS)) / denom;
            std::double_t y = (line1.rho*cos(line2.theta*DEGREES_TO_RADIANS) - line2.rho*cos(line1.theta*DEGREES_TO_RADIANS)) / denom;

            x += width * 0.5;
            y += height * 0.5;

            if (!(x < 0 || x > width || y < 0 || y > height))
            {
                intersections.emplace_back(x, y);
            }
        }
    }
    return intersections;
}

// Function to find endpoints of a line
std::vector<std::vector<cv::Point>> findPoints(const kht::ListOfLines& lines, std::int32_t width, std::int32_t height, std::int32_t relevant_lines)
{
    std::vector<std::vector<cv::Point>> points;
    double x1, x2, y1, y2;

    for (std::size_t j = 0; j != relevant_lines; ++j) 
    {
        auto const &line = lines[j];
        std::double_t rho = line.rho;
        std::double_t theta = line.theta * DEGREES_TO_RADIANS;
        std::double_t cos_theta = cos(theta), sin_theta = sin(theta);

        if (sin_theta != 0.0) 
        {
            x1 = -width * 0.5; y1 = (rho - x1 * cos_theta) / sin_theta;
            x2 = width * 0.5 - 1; y2 = (rho - x2 * cos_theta) / sin_theta;
        }
        else
        {
            x1 = rho; y1 = -height * 0.5;
            x2 = rho; y2 = height * 0.5 - 1;
        }

        x1 += width * 0.5; y1 += height * 0.5;
        x2 += width * 0.5; y2 += height * 0.5;

        std::vector<cv::Point> endpoints = {cv::Point(x1,y1), cv::Point(x2,y2)};
        points.push_back(endpoints);
    }
    return points;
}

// Function to find quadrilateral shapes from intersections
std::vector<std::vector<cv::Point>> findQuadrilaterals(const std::vector<cv::Point2f>& intersections)
{
    std::vector<std::vector<cv::Point>> quadrilaterals;
    for (std::size_t i = 0; i < intersections.size(); ++i)
    {
        for (std::size_t j = 0; j < intersections.size(); ++j) 
        {
            for (std::size_t k = 0; k < intersections.size(); ++k) 
            {
                for (std::size_t l = 0; l < intersections.size(); ++l) 
                {
                    std::vector<cv::Point> quadCandidate = {intersections[i], intersections[j], intersections[k], intersections[l]};     

                    std::vector<double> sides;
                    bool correct = true;
                    for (int i = 0; i < 4; ++i)
                    {
                        cv::Point diff = quadCandidate[i] - quadCandidate[(i + 1) % 4];
                        if (cv::norm(diff) < 10) {correct = false;}
                        sides.push_back(cv::norm(diff));
                    }

                    if (!(std::abs(sides[0] - sides[2]) > 10 || std::abs(sides[1] - sides[3]) > 10))
                    {
                        if (correct & cv::isContourConvex(quadCandidate)) {quadrilaterals.push_back(quadCandidate);} 
                    }                       
                }
            }
        }
    }
    return quadrilaterals;
}

int main(int argc, char* argv[])
{
    char window_name[512];

    cv::Mat im, gray, bw;
    kht::ListOfLines lines;
    cv::Scalar blue(255, 0, 0);
    cv::Scalar green(0, 255, 0);
    cv::Scalar red(0, 0, 255);

    std::string videoPath = "/home/caspl202/kht/extra/video.mp4";

    cv::VideoCapture videoCapture(videoPath);
    if (!videoCapture.isOpened()) 
    {
        std::cout << "Error opening video file." << std::endl;
        return EXIT_FAILURE;
    }

    int totalFrames = videoCapture.get(cv::CAP_PROP_FRAME_COUNT); 
    int framesToCapture = 10; 

    // Use a random number generator to get different frames
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, totalFrames - 1);

    for (int i = 0; i < framesToCapture; ++i) 
    {  
        // Generate a random frame index
        int frameIndex = dis(gen);

        // Set the frame position to the random index
        videoCapture.set(cv::CAP_PROP_POS_FRAMES, frameIndex);

        // Capture the frame at the random index
        if (!videoCapture.read(im)) {break;}

        // Save the current frame as an image file.
        std::string imagePath = "../../../extra/image_" + std::to_string(i) + ".jpg";
        cv::imwrite(imagePath, im);
        std::int32_t height = im.rows, width = im.cols;

        // Convert the input image to a binary edge image.
        cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, bw, 80, 200);

        // Call the kernel-base Hough transform function.
        kht::run_kht(lines, bw.ptr(), width, height);

        // Show current image and its most relevant detected lines.
        std::int32_t relevant_lines = 8;
        sprintf(window_name, "KHT - Image '%s' - %d most relevant lines", imagePath.c_str(), relevant_lines);
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

        // Find endpoints of lines
        std::vector<std::vector<cv::Point>> points = findPoints(lines, width, height, relevant_lines);

        // Draw lines on the image
        for (const auto& endpoints : points)
        {
            cv::line(im, endpoints[0], endpoints[1], green, 2, cv::LINE_8);
        }

        // Find intersections of lines
        std::vector<cv::Point2f> intersections = findIntersections(lines, width, height, relevant_lines);

        cv::Mat data_mat(intersections.size(), 2, CV_32FC1);

        for (size_t i = 0; i < intersections.size(); ++i)
        {
            data_mat.at<float>(i, 0) = intersections[i].x;
            data_mat.at<float>(i, 1) = intersections[i].y;
        }

        int num_clusters = 4; // number of clusters

        if (intersections.size() >= num_clusters) 
        {
            cv::Mat labels, centers;
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1.0);
            cv::kmeans(data_mat, num_clusters, labels, criteria, 3, cv::KMEANS_RANDOM_CENTERS, centers);

            std::vector<std::vector<cv::Point2f>> clustered_points(num_clusters);
            for (size_t i = 0; i < intersections.size(); ++i) 
            {
                int cluster_index = labels.at<int>(i, 0);
                clustered_points[cluster_index].push_back(intersections[i]);
            }
            
            std::vector<cv::Point2f> cluster_centers;
            for (int i = 0; i < num_clusters; ++i) 
            {
                cluster_centers.push_back(cv::Point2f(centers.at<float>(i, 0), centers.at<float>(i, 1)));
            }

            intersections = cluster_centers;
        }
        
        for (const auto& intersection : intersections) 
        {
            cv::circle(im, intersection, 1, red, 2, cv::LINE_8);
        }

        // Find quadrilateral shapes from intersections
        std::vector<std::vector<cv::Point>> quadrilaterals = findQuadrilaterals(intersections);

        // Draw quadrilaterals on the image
        for (const auto& quad : quadrilaterals) 
        {
            cv::polylines(im, quad, true, blue, 2, cv::LINE_8);
        }

        cv::imshow(window_name, im);
    }

    cv::waitKey(0);
    return EXIT_SUCCESS;
}




