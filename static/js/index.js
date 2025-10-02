window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 2,
			loop: true,
			infinite: true,
			autoplay: false,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    // Set video playback rate
    var comparisonVideo2 = document.getElementById('comparision-video-2');
    if (comparisonVideo2) {
        comparisonVideo2.addEventListener('loadedmetadata', function() {
            this.playbackRate = 4; // Set the playback rate to 4x
        });
    }

})

// Move this inside the document ready function

window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  // Only preload if the interpolation elements exist
  var interpolationWrapper = document.getElementById('interpolation-image-wrapper');
  if (!interpolationWrapper) {
    return;
  }
  
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var interpolationWrapper = document.getElementById('interpolation-image-wrapper');
  if (!interpolationWrapper) {
    return;
  }
  
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    console.log('Document ready - second function');
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    // Code for the interpolation slider control.
    var interpolationSlider = document.getElementById('interpolation-slider');
    if (interpolationSlider) {
      interpolationSlider.oninput = function() {
        setInterpolationImage(this.value);
      };
      setInterpolationImage(0);
      $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
    }

    // Only attach bulmaSlider if slider elements exist
    var sliderElements = document.querySelectorAll('.slider');
    if (sliderElements.length > 0) {
      bulmaSlider.attach();
    }

    // Code for the libero videos dropdown toggle.
    var liberoVideosToggleButton = document.querySelector('.toggle-libero-videos');
    var liberoVideosContent = document.getElementById('libero-videos-content');

    if (liberoVideosToggleButton && liberoVideosContent) {
      liberoVideosToggleButton.addEventListener('click', function() {
        if (liberoVideosContent.classList.contains('is-hidden')) {
          liberoVideosContent.classList.remove('is-hidden');
          liberoVideosToggleButton.querySelector('span:first-child').textContent = 'Hide Videos';
          liberoVideosToggleButton.querySelector('.icon i').classList.remove('fa-angle-down');
          liberoVideosToggleButton.querySelector('.icon i').classList.add('fa-angle-up');
        } else {
          liberoVideosContent.classList.add('is-hidden');
          liberoVideosToggleButton.querySelector('span:first-child').textContent = 'Show Videos';
          liberoVideosToggleButton.querySelector('.icon i').classList.remove('fa-angle-up');
          liberoVideosToggleButton.querySelector('.icon i').classList.add('fa-angle-down');
        }
      });
    }

    // Code for the three-tier dropdown system
    console.log('Setting up three-tier dropdown system...');
    
    // Add a small delay to ensure DOM is fully ready
    setTimeout(function() {
      var datasetSelector = document.getElementById('dataset-selector');
      var taskSelector = document.getElementById('task-selector');
      var videoSelector = document.getElementById('video-selector');
      var demoVideo = document.getElementById('demo-video');

      console.log('Elements found:', {
        datasetSelector: !!datasetSelector,
        taskSelector: !!taskSelector,
        videoSelector: !!videoSelector,
        demoVideo: !!demoVideo
      });

    if (datasetSelector && taskSelector && videoSelector && demoVideo) {
      console.log('All dropdown elements found successfully');
      console.log('Dataset selector:', datasetSelector);
      console.log('Task selector:', taskSelector);
      console.log('Video selector:', videoSelector);
      console.log('Demo video:', demoVideo);
      // Define the task structure for each dataset
      var taskStructure = {
        'libero': {
          'libero_90': [
            'KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.mp4',
            'KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.mp4',
            'KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet.mp4',
            'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.mp4',
            'KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it.mp4',
            'KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.mp4',
            'KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it.mp4',
            'KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate.mp4',
            'KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate.mp4',
            'KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.mp4',
            'KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet.mp4',
            'KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.mp4',
            'KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl.mp4',
            'KITCHEN_SCENE3_put_the_frying_pan_on_the_stove.mp4',
            'KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.mp4',
            'KITCHEN_SCENE3_turn_on_the_stove.mp4',
            'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it.mp4',
            'KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.mp4',
            'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet.mp4',
            'KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.mp4',
            'KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE5_put_the_black_bowl_on_the_plate.mp4',
            'KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet.mp4',
            'KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet.mp4',
            'KITCHEN_SCENE6_close_the_microwave.mp4',
            'KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.mp4',
            'KITCHEN_SCENE7_open_the_microwave.mp4',
            'KITCHEN_SCENE7_put_the_white_bowl_on_the_plate.mp4',
            'KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate.mp4',
            'KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove.mp4',
            'KITCHEN_SCENE8_turn_off_the_stove.mp4',
            'KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf.mp4',
            'KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet.mp4',
            'KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf.mp4',
            'KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet.mp4',
            'KITCHEN_SCENE9_turn_on_the_stove.mp4',
            'KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it.mp4',
            'LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.mp4',
            'LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.mp4',
            'LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray.mp4',
            'LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray.mp4',
            'LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray.mp4',
            'LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.mp4',
            'LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.mp4',
            'LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray.mp4',
            'LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray.mp4',
            'LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.mp4',
            'LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray.mp4',
            'LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate.mp4',
            'LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate.mp4',
            'LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate.mp4',
            'LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate.mp4',
            'LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate.mp4',
            'LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate.mp4',
            'LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate.mp4',
            'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate.mp4',
            'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.mp4',
            'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.mp4',
            'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.mp4',
            'STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy.mp4',
            'STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4',
            'STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.mp4',
            'STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.mp4',
            'STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.mp4',
            'STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.mp4',
            'STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.mp4',
            'STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.mp4',
            'STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy.mp4',
            'STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy.mp4',
            'STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf.mp4',
            'STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.mp4',
            'STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf.mp4',
            'STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf.mp4'
          ],
          'libero_long': [
            'close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.mp4',
            'pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.mp4',
            'pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf.mp4',
            'put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it.mp4',
            'put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.mp4',
            'put_the_chocolate_pudding_to_the_right_of_the_plate.mp4',
            'put_the_yellow_and_white_mug_on_the_right_plate.mp4',
            'stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.mp4',
            'close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.mp4',
            'stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray.mp4'
          ]
        }
      };

      // Function to populate task selector
      function populateTaskSelector(dataset) {
        console.log('Populating task selector for dataset:', dataset);
        taskSelector.innerHTML = '<option value="">Choose task type...</option>';
        videoSelector.innerHTML = '<option value="">Choose video...</option>';
        
        if (dataset && taskStructure[dataset]) {
          console.log('Found tasks for dataset:', Object.keys(taskStructure[dataset]));
          for (var task in taskStructure[dataset]) {
            var option = document.createElement('option');
            option.value = task;
            option.textContent = task.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            taskSelector.appendChild(option);
          }
          taskSelector.disabled = false;
          videoSelector.disabled = true;
          console.log('Task selector enabled, video selector disabled');
        } else {
          taskSelector.disabled = true;
          videoSelector.disabled = true;
          console.log('Both selectors disabled - no dataset or no tasks found');
        }
      }

      // Function to populate video selector
      function populateVideoSelector(dataset, task) {
        console.log('Populating video selector for dataset:', dataset, 'task:', task);
        videoSelector.innerHTML = '<option value="">Choose video...</option>';
        
        if (dataset && task && taskStructure[dataset] && taskStructure[dataset][task]) {
          var videos = taskStructure[dataset][task];
          console.log('Found videos for task:', videos);
          for (var i = 0; i < videos.length; i++) {
            var option = document.createElement('option');
            option.value = videos[i];
            // Format video names for display
            var displayName = videos[i].replace(/\.mp4/g, '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            // Remove common prefixes for better readability
            displayName = displayName.replace(/^Kitchen Scene\d+ /i, 'Kitchen: ');
            displayName = displayName.replace(/^Living Room Scene\d+ /i, 'Living Room: ');
            displayName = displayName.replace(/^Study Scene\d+ /i, 'Study: ');
            option.textContent = displayName;
            videoSelector.appendChild(option);
          }
          videoSelector.disabled = false;
          console.log('Video selector enabled with', videos.length, 'videos');
        } else {
          videoSelector.disabled = true;
          console.log('Video selector disabled - no videos found for this task');
        }
      }

      // Function to load and play video
      function loadVideo(videoFile) {
        console.log('Loading video:', videoFile);
        if (videoFile) {
          var videoSource = demoVideo.querySelector('source');
          var selectedDataset = datasetSelector.value;
          var selectedTask = taskSelector.value;
          
          console.log('Selected dataset:', selectedDataset, 'Selected task:', selectedTask);
          
          if (selectedDataset === 'libero') {
            if (selectedTask === 'libero_90') {
              videoSource.src = './static/videos/libero/libero_90/' + videoFile;
            } else if (selectedTask === 'libero_object') {
              videoSource.src = './static/videos/libero/libero_object/' + videoFile;
            } else if (selectedTask === 'libero_goal') {
              videoSource.src = './static/videos/libero/libero_goal/videos/' + videoFile;
            } else if (selectedTask === 'libero_long') {
              videoSource.src = './static/videos/libero/libero_long/' + videoFile;
            }
          } else if (selectedDataset === 'metaworld') {
            videoSource.src = './static/videos/metaworld/tasks/' + videoFile;
          } else if (selectedDataset === 'robocasa') {
            videoSource.src = './static/videos/robocasa/' + videoFile;
          }
          
          console.log('Video source set to:', videoSource.src);
          demoVideo.load();
        }
      }

      // Dataset selector event listener
      datasetSelector.addEventListener('change', function() {
        var selectedDataset = this.value;
        console.log('Dataset selected:', selectedDataset);
        populateTaskSelector(selectedDataset);
        demoVideo.pause();
      });

      // Task selector event listener
      taskSelector.addEventListener('change', function() {
        var selectedDataset = datasetSelector.value;
        var selectedTask = this.value;
        console.log('Task selected:', selectedTask);
        populateVideoSelector(selectedDataset, selectedTask);
        demoVideo.pause();
      });

      // Video selector event listener
      videoSelector.addEventListener('change', function() {
        var selectedVideo = this.value;
        console.log('Video selected:', selectedVideo);
        loadVideo(selectedVideo);
      });

      // Auto-select Libero dataset and libero_90 task on page load
      console.log('Initializing dropdowns...');
      try {
        datasetSelector.value = 'libero';
        populateTaskSelector('libero');
        taskSelector.value = 'libero_90';
        populateVideoSelector('libero', 'libero_90');
        videoSelector.value = 'KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.mp4';
        loadVideo('KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.mp4');
        console.log('Dropdowns initialized successfully');
      } catch (error) {
        console.error('Error initializing dropdowns:', error);
      }
    } else {
      console.error('Some dropdown elements not found');
    }
    }, 100); // End of setTimeout

});
