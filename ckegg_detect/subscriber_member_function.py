import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    # 初始化 ROS2 运行时（移除 with 语句）
    rclpy.init(args=args)
    
    # 创建订阅者节点
    minimal_subscriber = MinimalSubscriber()

    try:
        # 运行节点
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, ExternalShutdownException):
        # 捕获 Ctrl+C 或外部关闭信号
        pass
    finally:
        # 确保资源被正确清理
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()