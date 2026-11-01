# R3 FINAL DECISION: NO-GO

**SCIENTIFIC CONCLUSION: NO-GO. 192/192 run hợp lệ.**

| Budget | Arm | Mean length | Median instance length | 95% CI mean length | Iterations hợp lệ | Search s |
|---|---|---:|---:|---|---:|---:|
| iterations | K32 | 23.32759313 | 23.33130327 | [23.239937, 23.423771] | 1000.00 | 2.1988 |
| iterations | K40 | 23.33124896 | 23.33236185 | [23.250713, 23.422300] | 1000.00 | 2.2481 |
| iterations | Adaptive K32/K64 | 23.32192952 | 23.32857411 | [23.237078, 23.419503] | 1000.00 | 2.2820 |
| wall | K32 | 23.27530808 | 23.28156149 | [23.187679, 23.371131] | 3402.03 | 8.0172 |
| wall | K40 | 23.28380417 | 23.28145030 | [23.202868, 23.373771] | 3387.97 | 8.0100 |
| wall | Adaptive K32/K64 | 23.27823256 | 23.28627621 | [23.193688, 23.372720] | 3301.94 | 8.0153 |

| Budget | So sánh | Mean gain | Median gain | 95% CI mean | + / − / = instances | Worst | Best |
|---|---|---:|---:|---|---|---:|---:|
| iterations | C_A | 0.0240% | -0.0116% | [-0.0730%, 0.1333%] | 7/9/0 | -0.2568% | 0.6102% |
| iterations | C_B | 0.0401% | 0.0818% | [-0.0303%, 0.1071%] | 10/6/0 | -0.2056% | 0.2318% |
| wall | C_A | -0.0128% | -0.0199% | [-0.0754%, 0.0533%] | 7/9/0 | -0.2320% | 0.2896% |
| wall | C_B | 0.0241% | 0.0487% | [-0.0339%, 0.0832%] | 9/7/0 | -0.1539% | 0.2071% |

| Primary 8 s | Median ≥0.1% | Positive ≥12/16 | Bootstrap lower >0 |
|---|---|---|---|
| C_A | -0.0199% (FAIL) | 7/16 (FAIL) | -0.0754% (FAIL) |
| C_B | 0.0487% (FAIL) | 9/16 (FAIL) | -0.0339% (FAIL) |

Dừng đầu tư thêm vào đúng adaptive candidate policy K32/K64 này trên uniform TSP1K. Hướng ưu tiên tiếp theo là nghiên cứu SRR scheduling/revisit với equal-compute control và dữ liệu mới; M1 v2 có tín hiệu cơ học trực tiếp hơn nhưng chưa chứng minh hiệu quả end-to-end.

Protocol `e23677b760adffabfb06ff4ad03057c3a2d33b684d5a83feb48a98e1883f62cd`; dataset `a9f453a20ce225308ed9282302c3651489bc6f0487990a66a0f2ae7d9618848f`.
Nguồn thực thi `67423537c580717a8407612f0531a92f738e9f53`; Stage A giữ nguyên NO-GO.
Xem FINAL_RESEARCH_REPORT.md và analysis/decision.json; không có missing/imputed run.
