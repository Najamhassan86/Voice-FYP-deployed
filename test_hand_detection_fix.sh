#!/bin/bash

# Quick Fix Test Script for PSL Recognition
# Tests the hand detection validation

echo "🔍 Testing PSL Recognition Hand Detection Validation"
echo "=================================================="

API_URL="http://localhost:8000/api/psl/recognize"

# Generate test sequence (60 frames of 188 features)
echo "📝 Creating test sequence..."
python3 << 'EOF'
import json
import sys

# Create dummy sequence: 60 frames, 188 features each
sequence = [[0.0] * 188 for _ in range(60)]

# Test 1: No hands detected (should FAIL)
print("\n✅ TEST 1: No Hands Detected (Should REJECT)")
print("=" * 50)
request1 = {
    "sequence": sequence,
    "hands_detected": 0  # NO HANDS
}

import subprocess
result = subprocess.run(
    ['curl', '-s', '-X', 'POST', 
     'http://localhost:8000/api/psl/recognize',
     '-H', 'Content-Type: application/json',
     '-d', json.dumps(request1)],
    capture_output=True,
    text=True
)

response = json.loads(result.stdout)
print("Request:", f"{{'hands_detected': 0}}")
print("Response Status:", "ERROR (Expected ✅)" if 'detail' in response else "SUCCESS (Unexpected ❌)")
if 'detail' in response:
    print("Error Message:", response['detail'])
print()

# Test 2: One hand detected (should PASS)
print("\n✅ TEST 2: One Hand Detected (Should WORK)")
print("=" * 50)
request2 = {
    "sequence": sequence,
    "hands_detected": 1  # ONE HAND
}

result = subprocess.run(
    ['curl', '-s', '-X', 'POST', 
     'http://localhost:8000/api/psl/recognize',
     '-H', 'Content-Type: application/json',
     '-d', json.dumps(request2)],
    capture_output=True,
    text=True
)

response = json.loads(result.stdout)
print("Request:", f"{{'hands_detected': 1}}")
print("Response Status:", "SUCCESS ✅" if 'label' in response else "ERROR (Unexpected)")
if 'label' in response:
    print("Predicted Label:", response['label'])
    print("Confidence:", f"{response['confidence']:.2%}")
elif 'detail' in response:
    print("Error:", response['detail'])
print()

# Test 3: Two hands detected (should PASS)
print("\n✅ TEST 3: Two Hands Detected (Should WORK)")
print("=" * 50)
request3 = {
    "sequence": sequence,
    "hands_detected": 2  # TWO HANDS
}

result = subprocess.run(
    ['curl', '-s', '-X', 'POST', 
     'http://localhost:8000/api/psl/recognize',
     '-H', 'Content-Type: application/json',
     '-d', json.dumps(request3)],
    capture_output=True,
    text=True
)

response = json.loads(result.stdout)
print("Request:", f"{{'hands_detected': 2}}")
print("Response Status:", "SUCCESS ✅" if 'label' in response else "ERROR (Unexpected)")
if 'label' in response:
    print("Predicted Label:", response['label'])
    print("Confidence:", f"{response['confidence']:.2%}")
elif 'detail' in response:
    print("Error:", response['detail'])

print("\n" + "=" * 50)
print("✅ All validation tests completed!")
print("=" * 50)
EOF
