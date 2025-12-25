"""
Agent间通信机制
"""
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
import threading
import queue
from sqlalchemy.orm import Session

from storage.database.db import get_session
from storage.database.shared.academic_schema import TaskLog


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class Priority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AgentMessage:
    """Agent消息"""
    id: str
    sender: str
    receiver: str
    message_type: MessageType
    priority: Priority
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """从字典创建消息"""
        return cls(
            id=data["id"],
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=MessageType(data["message_type"]),
            priority=Priority(data["priority"]),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
        )


class MessageBus:
    """消息总线"""
    
    def __init__(self):
        self.queues = {}  # 每个Agent的消息队列
        self.subscribers = {}  # 事件订阅者
        self.message_handlers = {}  # 消息处理器
        self.lock = threading.RLock()
        self.running = False
        self.worker_thread = None
        
    def register_agent(self, agent_name: str):
        """注册Agent"""
        with self.lock:
            if agent_name not in self.queues:
                self.queues[agent_name] = queue.PriorityQueue()
                print(f"Agent {agent_name} 已注册到消息总线")
    
    def unregister_agent(self, agent_name: str):
        """注销Agent"""
        with self.lock:
            if agent_name in self.queues:
                del self.queues[agent_name]
                print(f"Agent {agent_name} 已从消息总线注销")
    
    def subscribe(self, event_type: str, handler: Callable[[AgentMessage], None]):
        """订阅事件"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def register_handler(self, agent_name: str, handler: Callable[[AgentMessage], AgentMessage]):
        """注册消息处理器"""
        self.message_handlers[agent_name] = handler
    
    def send_message(self, message: AgentMessage):
        """发送消息"""
        with self.lock:
            if message.receiver not in self.queues:
                raise ValueError(f"接收者 {message.receiver} 未注册")
            
            # 检查消息是否过期
            if message.expires_at and datetime.now() > message.expires_at:
                print(f"消息 {message.id} 已过期，丢弃")
                return
            
            # 将消息放入接收者队列（优先级队列）
            priority_item = (-message.priority.value, message.timestamp, message)
            self.queues[message.receiver].put(priority_item)
            
            # 记录消息
            self._log_message(message)
            
            # 触发事件订阅者
            event_type = f"message.{message.message_type.value}"
            if event_type in self.subscribers:
                for handler in self.subscribers[event_type]:
                    try:
                        handler(message)
                    except Exception as e:
                        print(f"事件处理器执行失败: {str(e)}")
    
    def receive_message(self, agent_name: str, timeout: float = None) -> Optional[AgentMessage]:
        """接收消息"""
        if agent_name not in self.queues:
            raise ValueError(f"Agent {agent_name} 未注册")
        
        try:
            priority_item = self.queues[agent_name].get(timeout=timeout)
            message = priority_item[2]  # 提取消息对象
            
            # 检查消息是否过期
            if message.expires_at and datetime.now() > message.expires_at:
                return None
            
            return message
            
        except queue.Empty:
            return None
    
    def broadcast(self, sender: str, message_type: MessageType, payload: Dict[str, Any], priority: Priority = Priority.NORMAL):
        """广播消息给所有Agent（除了发送者）"""
        with self.lock:
            for agent_name in self.queues:
                if agent_name != sender:
                    message = AgentMessage(
                        id=str(uuid.uuid4()),
                        sender=sender,
                        receiver=agent_name,
                        message_type=message_type,
                        priority=priority,
                        payload=payload,
                        timestamp=datetime.now()
                    )
                    self.send_message(message)
    
    def request_response(self, sender: str, receiver: str, payload: Dict[str, Any], timeout: float = 30.0) -> Optional[AgentMessage]:
        """发送请求并等待响应"""
        correlation_id = str(uuid.uuid4())
        
        # 发送请求消息
        request_message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            message_type=MessageType.REQUEST,
            priority=Priority.NORMAL,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        
        self.send_message(request_message)
        
        # 等待响应
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            response = self.receive_message(sender, timeout=1.0)
            if response and response.correlation_id == correlation_id:
                return response
        
        return None  # 超时
    
    def start_worker(self):
        """启动工作线程"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("消息总线工作线程已启动")
    
    def stop_worker(self):
        """停止工作线程"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        print("消息总线工作线程已停止")
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                # 处理所有Agent的消息
                with self.lock:
                    for agent_name, agent_queue in self.queues.items():
                        if agent_name in self.message_handlers:
                            # 处理该Agent的消息
                            while not agent_queue.empty():
                                priority_item = agent_queue.get()
                                message = priority_item[2]
                                
                                # 检查消息是否过期
                                if message.expires_at and datetime.now() > message.expires_at:
                                    continue
                                
                                # 调用消息处理器
                                try:
                                    handler = self.message_handlers[agent_name]
                                    response = handler(message)
                                    
                                    # 如果是请求消息，自动发送响应
                                    if message.message_type == MessageType.REQUEST and response:
                                        self.send_message(response)
                                        
                                except Exception as e:
                                    print(f"处理消息时出错: {str(e)}")
                                    
                                    # 发送错误响应
                                    if message.message_type == MessageType.REQUEST:
                                        error_response = AgentMessage(
                                            id=str(uuid.uuid4()),
                                            sender=agent_name,
                                            receiver=message.sender,
                                            message_type=MessageType.ERROR,
                                            priority=Priority.NORMAL,
                                            payload={"error": str(e)},
                                            timestamp=datetime.now(),
                                            correlation_id=message.correlation_id,
                                            reply_to=message.id
                                        )
                                        self.send_message(error_response)
                
                time.sleep(0.1)  # 避免CPU占用过高
                
            except Exception as e:
                print(f"工作线程出错: {str(e)}")
                time.sleep(1)
    
    def _log_message(self, message: AgentMessage):
        """记录消息到数据库"""
        try:
            with get_session() as session:
                task_log = TaskLog(
                    task_id=message.id,
                    agent_name=message.sender,
                    task_type=f"message.{message.message_type.value}",
                    status="started",
                    start_time=message.timestamp,
                    input_data=message.to_dict(),
                    metadata={
                        "receiver": message.receiver,
                        "priority": message.priority.value,
                        "correlation_id": message.correlation_id
                    }
                )
                session.add(task_log)
                session.commit()
        except Exception as e:
            print(f"记录消息失败: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取消息统计信息"""
        with self.lock:
            stats = {
                "registered_agents": list(self.queues.keys()),
                "queue_sizes": {},
                "total_handlers": len(self.message_handlers),
                "total_subscribers": sum(len(subs) for subs in self.subscribers.values())
            }
            
            for agent_name, agent_queue in self.queues.items():
                stats["queue_sizes"][agent_name] = agent_queue.qsize()
            
            return stats


# 全局消息总线实例
_message_bus = MessageBus()

def get_message_bus() -> MessageBus:
    """获取全局消息总线实例"""
    return _message_bus


class AgentCommunicator:
    """Agent通信器"""
    
    def __init__(self, agent_name: str, message_bus: MessageBus = None):
        self.agent_name = agent_name
        self.message_bus = message_bus or get_message_bus()
        self.handlers = {}
        
        # 注册到消息总线
        self.message_bus.register_agent(agent_name)
        
        # 注册默认处理器
        self.message_bus.register_handler(agent_name, self._handle_message)
    
    def __del__(self):
        """析构函数"""
        self.message_bus.unregister_agent(self.agent_name)
    
    def send(self, receiver: str, message_type: MessageType, payload: Dict[str, Any], 
             priority: Priority = Priority.NORMAL, correlation_id: str = None) -> str:
        """发送消息"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.agent_name,
            receiver=receiver,
            message_type=message_type,
            priority=priority,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        
        self.message_bus.send_message(message)
        return message.id
    
    def request(self, receiver: str, payload: Dict[str, Any], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """发送请求并等待响应"""
        response = self.message_bus.request_response(self.agent_name, receiver, payload, timeout)
        return response.payload if response else None
    
    def broadcast(self, message_type: MessageType, payload: Dict[str, Any], 
                  priority: Priority = Priority.NORMAL):
        """广播消息"""
        self.message_bus.broadcast(self.agent_name, message_type, payload, priority)
    
    def subscribe(self, event_type: str, handler: Callable[[AgentMessage], None]):
        """订阅事件"""
        self.message_bus.subscribe(event_type, handler)
    
    def register_handler(self, message_type: MessageType, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """注册特定消息类型的处理器"""
        self.handlers[message_type] = handler
    
    def receive(self, timeout: float = None) -> Optional[AgentMessage]:
        """接收消息"""
        return self.message_bus.receive_message(self.agent_name, timeout)
    
    def _handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理接收到的消息"""
        try:
            # 查找处理器
            if message.message_type in self.handlers:
                result_payload = self.handlers[message.message_type](message.payload)
                
                # 如果是请求消息，创建响应
                if message.message_type == MessageType.REQUEST:
                    response = AgentMessage(
                        id=str(uuid.uuid4()),
                        sender=self.agent_name,
                        receiver=message.sender,
                        message_type=MessageType.RESPONSE,
                        priority=Priority.NORMAL,
                        payload=result_payload,
                        timestamp=datetime.now(),
                        correlation_id=message.correlation_id,
                        reply_to=message.id
                    )
                    return response
            
            # 默认情况下，不发送响应
            return None
            
        except Exception as e:
            print(f"处理消息时出错: {str(e)}")
            
            # 发送错误响应
            if message.message_type == MessageType.REQUEST:
                error_response = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender=self.agent_name,
                    receiver=message.sender,
                    message_type=MessageType.ERROR,
                    priority=Priority.NORMAL,
                    payload={"error": str(e)},
                    timestamp=datetime.now(),
                    correlation_id=message.correlation_id,
                    reply_to=message.id
                )
                return error_response
            
            return None


# 初始化消息总线
def initialize_message_bus():
    """初始化消息总线"""
    message_bus = get_message_bus()
    message_bus.start_worker()
    return message_bus


# 停止消息总线
def shutdown_message_bus():
    """关闭消息总线"""
    message_bus = get_message_bus()
    message_bus.stop_worker()