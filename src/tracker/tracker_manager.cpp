#include "tracker_manager.h"

#include <string>

#include "helper/helper.h"
#include "train/tracker_trainer.h"

using std::string;

int ProcessTrackOutputJaccard(
		const BoundingBox& bbox_gt, BoundingBox* bbox_estimate,
		FILE* output_file_ptr);

double getJaccardCoefficient(double leftCol, double topRow, double rightCol, double bottomRow,
		double gtLeftCol, double gtTopRow, double gtRightCol, double gtBottomRow);


TrackerManager::TrackerManager(const std::vector<Video>& videos,
		RegressorBase* regressor, Tracker* tracker) :
		  videos_(videos),
		  regressor_(regressor),
		  tracker_(tracker)
{
}

void TrackerManager::TrackAll() {
	TrackAll(0, 1);
}

void TrackerManager::TrackAll(const size_t start_video_num, const int pause_val) {
	// Iterate over all videos and track the target object in each.
	FILE* output_file_ptr = fopen("/home/thomas/following-the-leader/GOTURN/result/result.txt", "w");

	for (size_t video_num = start_video_num; video_num < videos_.size(); ++video_num) {
		// Get the video.
		const Video& video = videos_[video_num];

		// Perform any pre-processing steps on this video.
		VideoInit(video, video_num);

		// Get the first frame of this video with the initial ground-truth bounding box (to initialize the tracker).
		int first_frame;
		cv::Mat image_curr;
		BoundingBox bbox_gt;
		video.LoadFirstAnnotation(&first_frame, &image_curr, &bbox_gt);

		// Initialize the tracker.
		tracker_->Init(image_curr, bbox_gt, regressor_);

		BoundingBox bbox_estimate_uncentered;

		int loss = false;
		double jaccard = 0.0;
		// Iterate over the remaining frames of the video.
		for (size_t frame_num = first_frame + 1; frame_num < video.all_frames.size(); ++frame_num) {

			// Get image for the current frame.
			// (The ground-truth bounding box is used only for visualization).
			const bool draw_bounding_box = false;
			const bool load_only_annotation = false;
			cv::Mat image_curr;
			BoundingBox bbox_gt;
			bool has_annotation = video.LoadFrame(frame_num,
					draw_bounding_box,
					load_only_annotation,
					&image_curr, &bbox_gt);

			// Get ready to track the object.
			SetupEstimate();

			// Track and estimate the target's bounding box location in the current image.
			// Important: this method cannot receive bbox_gt (the ground-truth bounding box) as an input.
			tracker_->Track(image_curr, regressor_, &bbox_estimate_uncentered, loss);

			// Process the output (e.g. visualize / save results).
			ProcessTrackOutput(frame_num, image_curr, has_annotation, bbox_gt,
					bbox_estimate_uncentered, pause_val);

			//calculate jaccard and print outputfile
			loss = ProcessTrackOutputJaccard(bbox_gt, &bbox_estimate_uncentered, output_file_ptr);


		}
		PostProcessVideo();
	}
	PostProcessAll();
}

TrackerVisualizer::TrackerVisualizer(const std::vector<Video>& videos,
		RegressorBase* regressor, Tracker* tracker) :
		  TrackerManager(videos, regressor, tracker)
{
}


void TrackerVisualizer::ProcessTrackOutput(
		const size_t frame_num, const cv::Mat& image_curr, const bool has_annotation,
		const BoundingBox& bbox_gt, const BoundingBox& bbox_estimate_uncentered,
		const int pause_val) {
	cv::Mat full_output;
	image_curr.copyTo(full_output);

	if (has_annotation) {
		// Draw ground-truth bounding box of the target location (white).
		bbox_gt.DrawBoundingBox(&full_output);
	}

	// Draw estimated bounding box of the target location (red).
	bbox_estimate_uncentered.Draw(255, 0, 0, &full_output);

	// Show the image with the estimated and ground-truth bounding boxes.
	cv::namedWindow("Full output", cv::WINDOW_AUTOSIZE ); // Create a window for display.
	cv::imshow("Full output", full_output );                   // Show our image inside it.

	// Pause for pause_val milliseconds, or until user input (if pause_val == 0).
	cv::waitKey(pause_val);
}

void TrackerVisualizer::VideoInit(const Video& video, const size_t video_num) {
	printf("Video: %zu\n", video_num);
}

double
getJaccardCoefficient(double leftCol, double topRow, double rightCol, double bottomRow,
		double gtLeftCol, double gtTopRow, double gtRightCol, double gtBottomRow)
{
	double jaccCoeff = 0.;

	if (!(leftCol > gtRightCol ||
			rightCol < gtLeftCol ||
			topRow > gtBottomRow ||
			bottomRow < gtTopRow)
	)
	{
		double interLeftCol = std::max<double>(leftCol, gtLeftCol);
		double interTopRow = std::max<double>(topRow, gtTopRow);
		double interRightCol = std::min<double>(rightCol, gtRightCol);
		double interBottomRow = std::min<double>(bottomRow, gtBottomRow);

		const double areaIntersection = (abs(interRightCol - interLeftCol) + 1) * (abs(interBottomRow - interTopRow) + 1);
		const double lhRoiSize = (abs(rightCol - leftCol) + 1) * (abs(bottomRow - topRow) + 1);
		const double rhRoiSize = (abs(gtRightCol - gtLeftCol) + 1) * (abs(gtBottomRow - gtTopRow) + 1);

		jaccCoeff = areaIntersection / (lhRoiSize + rhRoiSize - areaIntersection);
	}
	return jaccCoeff;
}



int ProcessTrackOutputJaccard(
		const BoundingBox& bbox_gt, BoundingBox* bbox_estimate,
		FILE* output_file_ptr) {

	int loss = 0;

	// Stop the timer and print the time needed for tracking.
	//  hrt_.stop();
	//  const double ms = hrt_.getMilliseconds();

	// Update the total time needed for tracking.  (Other time is used to save the tracking
	// output to a video and to write tracking data to a file for evaluation purposes).
	//  total_ms_ += ms;
	//  num_frames_++;

	// Get the tracking output.
	double width = fabs(bbox_estimate->get_width());
	double height = fabs(bbox_estimate->get_height());
	double x_min = std::min(bbox_estimate->x1_, bbox_estimate->x2_);
	double y_min = std::min(bbox_estimate->y1_, bbox_estimate->y2_);

	double width_gt = fabs(bbox_gt.get_width());
	double height_gt = fabs(bbox_gt.get_height());
	double x_min_gt = std::min(bbox_gt.x1_, bbox_gt.x2_);
	double y_min_gt = std::min(bbox_gt.y1_, bbox_gt.y2_);

	double l = x_min;
	double t = y_min;
	double r = x_min + width;
	double b = y_min + height;

	double l_gt = x_min_gt;
	double t_gt = y_min_gt;
	double r_gt = x_min_gt + width_gt;
	double b_gt = y_min_gt + height_gt;

	double jaccard = getJaccardCoefficient(l, t, r, b, l_gt, t_gt, r_gt, b_gt);


	if (jaccard < 0.2)
	{
		*bbox_estimate = bbox_gt;
		loss = 1;
	}
	else
	{
		loss = 0;
	}

	// Save the tracking output to a file
	fprintf(output_file_ptr, "%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%d\n", x_min, y_min, width,
			height,jaccard,loss);

	return loss;
}

TrackerTesterAlov::TrackerTesterAlov(const std::vector<Video>& videos,
		const bool save_videos,
		RegressorBase* regressor, Tracker* tracker,
		const std::string& output_folder) :
		  TrackerManager(videos, regressor, tracker),
		  output_folder_(output_folder),
		  hrt_("Tracker"),
		  total_ms_(0),
		  num_frames_(0),
		  save_videos_(save_videos)
{
}

void TrackerTesterAlov::VideoInit(const Video& video, const size_t video_num) {
	// Get the name of the video from the video file path.
	int delim_pos = video.path.find_last_of("/");
	const string& video_name = video.path.substr(delim_pos+1, video.path.length());
	printf("Video %zu: %s\n", video_num + 1, video_name.c_str());

	// Open a file for saving the tracking output.
	const string& output_file = output_folder_ + "/" + video_name;
	output_file_ptr_ = fopen(output_file.c_str(), "w");

	if (save_videos_) {
		// Make a folder to save the tracking videos.
		const string& video_out_folder = output_folder_ + "/videos";
		boost::filesystem::create_directories(video_out_folder);

		// Get the size of the images that will be saved.
		cv::Mat image;
		BoundingBox box;
		video.LoadFrame(0, false, false, &image, &box);

		// Open a video_writer object to save the tracking videos.
		const string video_out_name = video_out_folder + "/Video" + num2str(static_cast<int>(video_num)) + ".avi";
		video_writer_.open(video_out_name, CV_FOURCC('M','J','P','G'), 50, image.size());
	}
}

void TrackerTesterAlov::SetupEstimate() {
	// Record the time before starting to track.
	hrt_.reset();
	hrt_.start();
}

void TrackerTesterAlov::ProcessTrackOutput(
		const size_t frame_num, const cv::Mat& image_curr, const bool has_annotation,
		const BoundingBox& bbox_gt, const BoundingBox& bbox_estimate,
		const int pause_val) {
	// Stop the timer and print the time needed for tracking.
	hrt_.stop();
	const double ms = hrt_.getMilliseconds();

	// Update the total time needed for tracking.  (Other time is used to save the tracking
	// output to a video and to write tracking data to a file for evaluation purposes).
	total_ms_ += ms;
	num_frames_++;

	// Get the tracking output.
	const double width = fabs(bbox_estimate.get_width());
	const double height = fabs(bbox_estimate.get_height());
	const double x_min = std::min(bbox_estimate.x1_, bbox_estimate.x2_);
	const double y_min = std::min(bbox_estimate.y1_, bbox_estimate.y2_);

	// Save the trackign output to a file inthe appropriate format for the ALOV dataset.
	fprintf(output_file_ptr_, "%zu %lf %lf %lf %lf\n", frame_num + 1, x_min, y_min, width,
			height);

	if (save_videos_) {
		cv::Mat full_output;
		image_curr.copyTo(full_output);

		if (has_annotation) {
			// Draw ground-truth bounding box (white).
			bbox_gt.DrawBoundingBox(&full_output);
		}

		// Draw estimated bounding box on image (red).
		bbox_estimate.Draw(255, 0, 0, &full_output);

		// Save the image to a tracking video.
		video_writer_.write(full_output);
	}
}

void TrackerTesterAlov::PostProcessVideo() {
	// Close the file that saves the tracking data.
	fclose(output_file_ptr_);
}

void TrackerTesterAlov::PostProcessAll() {
	printf("Finished tracking %zu videos with %d total frames\n", videos_.size(), num_frames_);

	// Compute the mean tracking time per frame.
	const double mean_time_ms = total_ms_ / num_frames_;
	printf("Mean time: %lf ms\n", mean_time_ms);
}
