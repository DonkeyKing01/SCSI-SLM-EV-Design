"""
性能监控和数据验证模块
用于监控知识图谱构建过程的性能和验证数据质量
"""
import time
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
import os
import sys

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.neo4j_connector import neo4j_connector


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    operation: str
    duration: float
    memory_usage_mb: float
    cpu_percent: float
    records_processed: int = 0
    records_per_second: float = 0.0
    error_count: int = 0
    success: bool = True
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """数据验证结果"""
    check_name: str
    passed: bool
    expected_value: Any
    actual_value: Any
    error_message: str = ""
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化性能监控器"""
        self.metrics_history: List[PerformanceMetrics] = []
        self.validation_results: List[ValidationResult] = []
        self.start_time: Optional[datetime] = None
        self.memory_threshold_mb = 4096  # 4GB内存阈值
        self.cpu_threshold_percent = 80.0  # 80% CPU阈值
        
        # 性能统计
        self.stats = {
            'total_operations': 0,
            'total_duration': 0.0,
            'total_records': 0,
            'total_errors': 0,
            'peak_memory_mb': 0.0,
            'peak_cpu_percent': 0.0
        }
    
    def start_monitoring(self):
        """开始监控"""
        self.start_time = datetime.now()
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"性能监控已停止，总耗时: {total_time:.2f}秒")
            self._generate_performance_report()
    
    def monitor_operation(self, operation_name: str):
        """
        操作监控装饰器
        
        用法:
        @monitor.monitor_operation("创建节点")
        def create_nodes(self, nodes):
            # 具体操作
            pass
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                return self._execute_with_monitoring(operation_name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def _execute_with_monitoring(self, operation_name: str, func: Callable, *args, **kwargs):
        """执行带监控的操作"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        success = True
        error_count = 0
        records_processed = 0
        
        try:
            # 执行操作
            result = func(*args, **kwargs)
            
            # 尝试从结果中提取记录数量
            if isinstance(result, (list, tuple)):
                records_processed = len(result)
            elif isinstance(result, dict) and 'count' in result:
                records_processed = result['count']
            elif hasattr(result, '__len__'):
                records_processed = len(result)
                
            return result
            
        except Exception as e:
            success = False
            error_count = 1
            logger.error(f"操作 '{operation_name}' 执行失败: {e}")
            raise
            
        finally:
            # 计算性能指标
            end_time = time.time()
            duration = end_time - start_time
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            avg_memory = (start_memory + end_memory) / 2
            avg_cpu = (start_cpu + end_cpu) / 2
            
            records_per_second = records_processed / duration if duration > 0 else 0
            
            # 记录指标
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                operation=operation_name,
                duration=duration,
                memory_usage_mb=avg_memory,
                cpu_percent=avg_cpu,
                records_processed=records_processed,
                records_per_second=records_per_second,
                error_count=error_count,
                success=success
            )
            
            self._record_metrics(metrics)
            
            # 检查性能阈值
            self._check_performance_thresholds(metrics)
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """记录性能指标"""
        self.metrics_history.append(metrics)
        
        # 更新统计信息
        self.stats['total_operations'] += 1
        self.stats['total_duration'] += metrics.duration
        self.stats['total_records'] += metrics.records_processed
        self.stats['total_errors'] += metrics.error_count
        self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], metrics.memory_usage_mb)
        self.stats['peak_cpu_percent'] = max(self.stats['peak_cpu_percent'], metrics.cpu_percent)
        
        # 记录日志
        logger.info(
            f"操作完成: {metrics.operation} | "
            f"耗时: {metrics.duration:.2f}s | "
            f"记录数: {metrics.records_processed} | "
            f"速度: {metrics.records_per_second:.1f}条/秒 | "
            f"内存: {metrics.memory_usage_mb:.1f}MB | "
            f"CPU: {metrics.cpu_percent:.1f}%"
        )
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """检查性能阈值"""
        # 内存使用检查
        if metrics.memory_usage_mb > self.memory_threshold_mb:
            logger.warning(
                f"内存使用超过阈值: {metrics.memory_usage_mb:.1f}MB > {self.memory_threshold_mb}MB"
            )
        
        # CPU使用检查
        if metrics.cpu_percent > self.cpu_threshold_percent:
            logger.warning(
                f"CPU使用超过阈值: {metrics.cpu_percent:.1f}% > {self.cpu_threshold_percent}%"
            )
        
        # 处理速度检查 (如果速度过慢)
        if metrics.records_processed > 100 and metrics.records_per_second < 10:
            logger.warning(
                f"处理速度较慢: {metrics.records_per_second:.1f}条/秒 "
                f"(操作: {metrics.operation})"
            )
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量 (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def _generate_performance_report(self):
        """生成性能报告"""
        if not self.metrics_history:
            logger.info("没有性能数据可报告")
            return
        
        logger.info("=" * 60)
        logger.info("性能监控报告")
        logger.info("=" * 60)
        
        # 总体统计
        logger.info(f"总操作数: {self.stats['total_operations']}")
        logger.info(f"总耗时: {self.stats['total_duration']:.2f}秒")
        logger.info(f"总记录数: {self.stats['total_records']}")
        logger.info(f"总错误数: {self.stats['total_errors']}")
        logger.info(f"峰值内存: {self.stats['peak_memory_mb']:.1f}MB")
        logger.info(f"峰值CPU: {self.stats['peak_cpu_percent']:.1f}%")
        
        # 平均性能
        avg_duration = self.stats['total_duration'] / self.stats['total_operations']
        avg_speed = self.stats['total_records'] / self.stats['total_duration'] if self.stats['total_duration'] > 0 else 0
        
        logger.info(f"平均操作耗时: {avg_duration:.2f}秒")
        logger.info(f"平均处理速度: {avg_speed:.1f}条/秒")
        
        # 按操作类型统计
        operation_stats = {}
        for metrics in self.metrics_history:
            op = metrics.operation
            if op not in operation_stats:
                operation_stats[op] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'total_records': 0,
                    'errors': 0
                }
            
            operation_stats[op]['count'] += 1
            operation_stats[op]['total_duration'] += metrics.duration
            operation_stats[op]['total_records'] += metrics.records_processed
            operation_stats[op]['errors'] += metrics.error_count
        
        logger.info("\n操作类型统计:")
        for op, stats in operation_stats.items():
            avg_time = stats['total_duration'] / stats['count']
            avg_speed = stats['total_records'] / stats['total_duration'] if stats['total_duration'] > 0 else 0
            logger.info(f"  {op}:")
            logger.info(f"    次数: {stats['count']}")
            logger.info(f"    平均耗时: {avg_time:.2f}秒")
            logger.info(f"    平均速度: {avg_speed:.1f}条/秒")
            logger.info(f"    错误数: {stats['errors']}")
        
        logger.info("=" * 60)
    
    def save_metrics_to_file(self, filepath: str):
        """保存性能指标到文件"""
        try:
            metrics_data = []
            for metrics in self.metrics_history:
                metrics_data.append({
                    'timestamp': metrics.timestamp.isoformat(),
                    'operation': metrics.operation,
                    'duration': metrics.duration,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'cpu_percent': metrics.cpu_percent,
                    'records_processed': metrics.records_processed,
                    'records_per_second': metrics.records_per_second,
                    'error_count': metrics.error_count,
                    'success': metrics.success,
                    'additional_info': metrics.additional_info
                })
            
            report_data = {
                'summary': self.stats,
                'metrics': metrics_data,
                'validation_results': [
                    {
                        'check_name': vr.check_name,
                        'passed': vr.passed,
                        'expected_value': vr.expected_value,
                        'actual_value': vr.actual_value,
                        'error_message': vr.error_message,
                        'severity': vr.severity
                    }
                    for vr in self.validation_results
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"性能指标已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存性能指标失败: {e}")


class DataValidator:
    """数据验证器"""
    
    def __init__(self, neo4j_connector):
        """初始化数据验证器"""
        self.connector = neo4j_connector
        self.validation_results: List[ValidationResult] = []
    
    def validate_node_counts(self, expected_counts: Dict[str, int]) -> List[ValidationResult]:
        """验证节点数量"""
        results = []
        
        for node_label, expected_count in expected_counts.items():
            try:
                query = f"MATCH (n:{node_label}) RETURN count(n) as count"
                result = self.connector.execute_query(query)
                actual_count = result[0]['count'] if result else 0
                
                passed = actual_count == expected_count
                severity = "ERROR" if not passed else "INFO"
                
                validation_result = ValidationResult(
                    check_name=f"{node_label}节点数量检查",
                    passed=passed,
                    expected_value=expected_count,
                    actual_value=actual_count,
                    severity=severity
                )
                
                results.append(validation_result)
                
            except Exception as e:
                validation_result = ValidationResult(
                    check_name=f"{node_label}节点数量检查",
                    passed=False,
                    expected_value=expected_count,
                    actual_value=None,
                    error_message=str(e),
                    severity="CRITICAL"
                )
                results.append(validation_result)
        
        self.validation_results.extend(results)
        return results
    
    def validate_data_quality(self) -> List[ValidationResult]:
        """验证数据质量"""
        results = []
        
        # 检查项定义
        quality_checks = [
            {
                'name': '评论内容非空检查',
                'query': "MATCH (r:Review) WHERE r.content IS NULL OR r.content = '' RETURN count(r) as count",
                'expected': 0,
                'description': '评论内容不应为空'
            },
            {
                'name': '情感分数范围检查',
                'query': "MATCH (r:Review) WHERE r.overallSentiment < -1 OR r.overallSentiment > 1 RETURN count(r) as count",
                'expected': 0,
                'description': '情感分数应在-1到1范围内'
            },
            {
                'name': '车型名称唯一性检查',
                'query': "MATCH (c:CarModel) WITH c.name as name, count(*) as cnt WHERE cnt > 1 RETURN count(*) as count",
                'expected': 0,
                'description': '车型名称应该唯一'
            },
            {
                'name': '用户画像ID唯一性检查',
                'query': "MATCH (u:UserProfile) WITH u.profileId as id, count(*) as cnt WHERE cnt > 1 RETURN count(*) as count",
                'expected': 0,
                'description': '用户画像ID应该唯一'
            },
            {
                'name': '特征维度数量检查',
                'query': "MATCH (f:Feature) RETURN count(f) as count",
                'expected': 8,
                'description': '应该有8个特征维度'
            }
        ]
        
        for check in quality_checks:
            try:
                result = self.connector.execute_query(check['query'])
                actual_value = result[0]['count'] if result else None
                passed = actual_value == check['expected']
                
                severity = "ERROR" if not passed else "INFO"
                
                validation_result = ValidationResult(
                    check_name=check['name'],
                    passed=passed,
                    expected_value=check['expected'],
                    actual_value=actual_value,
                    error_message="" if passed else check['description'],
                    severity=severity
                )
                
                results.append(validation_result)
                
            except Exception as e:
                validation_result = ValidationResult(
                    check_name=check['name'],
                    passed=False,
                    expected_value=check['expected'],
                    actual_value=None,
                    error_message=f"查询执行失败: {e}",
                    severity="CRITICAL"
                )
                results.append(validation_result)
        
        self.validation_results.extend(results)
        return results
    
    def validate_relationships(self) -> List[ValidationResult]:
        """验证关系完整性"""
        results = []
        
        relationship_checks = [
            {
                'name': '评论-车型关系检查',
                'query': "MATCH (r:Review)-[:MENTIONS]->(c:CarModel) RETURN count(*) > 0 as exists",
                'expected': True,
                'description': '应该存在评论到车型的MENTIONS关系'
            },
            {
                'name': '评论-特征关系检查',
                'query': "MATCH (r:Review)-[:CONTAINS_ASPECT]->(f:Feature) RETURN count(*) > 0 as exists",
                'expected': True,
                'description': '应该存在评论到特征的CONTAINS_ASPECT关系'
            },
            {
                'name': '用户画像-评论关系检查',
                'query': "MATCH (u:UserProfile)-[:PUBLISHED]->(r:Review) RETURN count(*) > 0 as exists",
                'expected': True,
                'description': '应该存在用户画像到评论的PUBLISHED关系'
            },
            {
                'name': '用户画像-车型关系检查',
                'query': "MATCH (u:UserProfile)-[:INTERESTED_IN]->(c:CarModel) RETURN count(*) > 0 as exists",
                'expected': True,
                'description': '应该存在用户画像到车型的INTERESTED_IN关系'
            }
        ]
        
        for check in relationship_checks:
            try:
                result = self.connector.execute_query(check['query'])
                actual_value = result[0]['exists'] if result else False
                passed = actual_value == check['expected']
                
                severity = "ERROR" if not passed else "INFO"
                
                validation_result = ValidationResult(
                    check_name=check['name'],
                    passed=passed,
                    expected_value=check['expected'],
                    actual_value=actual_value,
                    error_message="" if passed else check['description'],
                    severity=severity
                )
                
                results.append(validation_result)
                
            except Exception as e:
                validation_result = ValidationResult(
                    check_name=check['name'],
                    passed=False,
                    expected_value=check['expected'],
                    actual_value=None,
                    error_message=f"查询执行失败: {e}",
                    severity="CRITICAL"
                )
                results.append(validation_result)
        
        self.validation_results.extend(results)
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        if not self.validation_results:
            return {"message": "没有验证结果"}
        
        # 统计结果
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results if result.passed)
        failed_checks = total_checks - passed_checks
        
        # 按严重级别分组
        severity_counts = {}
        failed_by_severity = []
        
        for result in self.validation_results:
            if not result.passed:
                failed_by_severity.append(result)
            
            if result.severity not in severity_counts:
                severity_counts[result.severity] = 0
            severity_counts[result.severity] += 1
        
        report = {
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'success_rate': round(passed_checks / total_checks * 100, 2) if total_checks > 0 else 0
            },
            'severity_distribution': severity_counts,
            'failed_checks': [
                {
                    'check_name': result.check_name,
                    'expected': result.expected_value,
                    'actual': result.actual_value,
                    'error': result.error_message,
                    'severity': result.severity
                }
                for result in failed_by_severity
            ],
            'all_results': [
                {
                    'check_name': result.check_name,
                    'passed': result.passed,
                    'expected': result.expected_value,
                    'actual': result.actual_value,
                    'severity': result.severity
                }
                for result in self.validation_results
            ]
        }
        
        return report


# 创建全局实例
performance_monitor = PerformanceMonitor()
data_validator = DataValidator(neo4j_connector)