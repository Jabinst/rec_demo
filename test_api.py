"""简单的 API 测试脚本，用于验证推荐服务是否正常工作。"""

import requests
import json


def test_recommend_api():
    """测试推荐接口。"""
    base_url = "http://127.0.0.1:8000"
    
    print("=" * 60)
    print("测试推荐系统 API")
    print("=" * 60)
    
    # 1. 测试健康检查
    print("\n[1] 测试健康检查接口...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"  状态码: {response.status_code}")
        print(f"  响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        print("  请确保服务已启动: uvicorn rec_demo.service:app --reload --port 8000")
        return
    
    # 2. 测试推荐接口
    print("\n[2] 测试推荐接口...")
    test_cases = [
        {"user_id": 1, "top_k": 3},
        {"user_id": 2, "top_k": 5},
    ]
    
    for case in test_cases:
        print(f"\n  测试用户 {case['user_id']} 的 Top-{case['top_k']} 推荐:")
        try:
            response = requests.post(
                f"{base_url}/api/recommend",
                json=case,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"    ✓ 成功获取 {len(result['recommendations'])} 条推荐")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"      {i}. {rec['title']} (分数: {rec['score']:.4f})")
            else:
                print(f"    ❌ 请求失败: {response.status_code}")
                print(f"    错误信息: {response.text}")
        except Exception as e:
            print(f"    ❌ 请求异常: {e}")
    
    # 3. 测试用户信息接口
    print("\n[3] 测试用户信息接口...")
    try:
        response = requests.get(f"{base_url}/api/user/1/info")
        if response.status_code == 200:
            print(f"  用户信息: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"  ❌ 请求失败: {response.status_code}")
    except Exception as e:
        print(f"  ❌ 请求异常: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_recommend_api()
