var ros = new ROSLIB.Ros({
    url : 'ws://' + address + ':' + port
  });

  ros.on('connection', function() {
    document.getElementById("status").innerHTML = "Connected";
  });

  ros.on('error', function(error) {
    document.getElementById("status").innerHTML = "Error";
  });

  ros.on('close', function() {
    document.getElementById("status").innerHTML = "Closed";
  });

  // Listen to /txt_msg: ------------------- //
  var txt_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/txt_msg',
    messageType : 'std_msgs/String'
  });

  // Listen to /cmd_vel: ------------------- //
  var cmd_vel_listener = new ROSLIB.Topic({
    ros : ros,
    name : "/jackal_velocity_controller/cmd_vel",
    messageType : 'geometry_msgs/Twist'
  });

  move = function (linear, angular) {
    var twist = new ROSLIB.Message({
      linear: {
        x: linear,
        y: 0,
        z: 0
      },
      angular: {
        x: 0,
        y: 0,
        z: angular
      }
    });
    cmd_vel_listener.publish(twist);
  }

  txt_listener.subscribe(function(m) {
    document.getElementById("msg").innerHTML = m.data;
  });

  createJoystick = function () {
    var options = {
      zone: document.getElementById('zone_joystick'),
      threshold: 0.1,
      position: { left: 50 + '%' },
      mode: 'static',
      size: 400,
      //color: '#000000',
      color: 'blue',
    };
    manager = nipplejs.create(options);

    linear_speed = 0;
    angular_speed = 0;

    self.manager.on('start', function (event, nipple) {
      timer = setInterval(function () {
        move(linear_speed, angular_speed);
      }, 25);
    });

    self.manager.on('move', function (event, nipple) {
      max_linear = 0.8; // m/s
      max_angular = 1.4; // rad/s
      max_distance = 400.0; // pixels;
      linear_speed = Math.sin(nipple.angle.radian) * max_linear * nipple.distance/max_distance;
      angular_speed = -Math.cos(nipple.angle.radian) * max_angular * nipple.distance/max_distance;
    });

    self.manager.on('end', function () {
      if (timer) {
        clearInterval(timer);
      }
      self.move(0, 0);
    });
  }
  window.onload = function () {
    createJoystick();
  }
