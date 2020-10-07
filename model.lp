\ Model TOP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  -0 y_1_1 + -0 u_1_1 + -0 x_(1,1)_1 + 14 y_2_1 + -0 x_(2,1)_1
   + -0 x_(2,2)_1 + 71 y_3_1 + -0 x_(3,1)_1 + -0 x_(3,3)_1 + 60 y_4_1
   + -0 x_(4,1)_1 + -0 x_(4,4)_1 + -0 y_5_1 + -0 x_(5,1)_1 + -0 x_(5,5)_1
Subject To
 must_visit_depot_first: x_(1,2)_1 + x_(1,3)_1 + x_(1,4)_1 + x_(1,5)_1 = 1
 must_visit_depot_end: x_(1,5)_1 + x_(2,5)_1 + x_(3,5)_1 + x_(4,5)_1 = 1
 only_one_visit_pr_scooter_(k=2): y_2_1 <= 1
 only_one_visit_pr_scooter_(k=3): y_3_1 <= 1
 only_one_visit_pr_scooter_(k=4): y_4_1 <= 1
 connectivity_1_(k=2,v=1): x_(1,2)_1 - y_2_1 + x_(3,2)_1 + x_(4,2)_1 = 0
 connectivity_2_(k=2,v=1): - y_2_1 + x_(2,3)_1 + x_(2,4)_1 + x_(2,5)_1 = 0
 connectivity_1_(k=3,v=1): x_(1,3)_1 + x_(2,3)_1 - y_3_1 + x_(4,3)_1 = 0
 connectivity_2_(k=3,v=1): - y_3_1 + x_(3,2)_1 + x_(3,4)_1 + x_(3,5)_1 = 0
 connectivity_1_(k=4,v=1): x_(1,4)_1 + x_(2,4)_1 + x_(3,4)_1 - y_4_1 = 0
 connectivity_2_(k=4,v=1): - y_4_1 + x_(4,2)_1 + x_(4,3)_1 + x_(4,5)_1 = 0
 subtours_1_(i=2,v=1): - u_2_1 <= -2
 subtours_2_(i=2,v=1): u_2_1 <= 5
 subtours_3_(i,j=(2,3),v=1): u_2_1 + 4 x_(2,3)_1 - u_3_1 <= 3
 subtours_3_(i,j=(2,4),v=1): u_2_1 + 4 x_(2,4)_1 - u_4_1 <= 3
 subtours_3_(i,j=(2,5),v=1): u_2_1 + 4 x_(2,5)_1 - u_5_1 <= 3
 subtours_1_(i=3,v=1): - u_3_1 <= -2
 subtours_2_(i=3,v=1): u_3_1 <= 5
 subtours_3_(i,j=(3,2),v=1): - u_2_1 + u_3_1 + 4 x_(3,2)_1 <= 3
 subtours_3_(i,j=(3,4),v=1): u_3_1 + 4 x_(3,4)_1 - u_4_1 <= 3
 subtours_3_(i,j=(3,5),v=1): u_3_1 + 4 x_(3,5)_1 - u_5_1 <= 3
 subtours_1_(i=4,v=1): - u_4_1 <= -2
 subtours_2_(i=4,v=1): u_4_1 <= 5
 subtours_3_(i,j=(4,2),v=1): - u_2_1 + u_4_1 + 4 x_(4,2)_1 <= 3
 subtours_3_(i,j=(4,3),v=1): - u_3_1 + u_4_1 + 4 x_(4,3)_1 <= 3
 subtours_3_(i,j=(4,5),v=1): u_4_1 + 4 x_(4,5)_1 - u_5_1 <= 3
 subtours_1_(i=5,v=1): - u_5_1 <= -2
 subtours_2_(i=5,v=1): u_5_1 <= 5
 subtours_3_(i,j=(5,2),v=1): - u_2_1 + u_5_1 + 4 x_(5,2)_1 <= 3
 subtours_3_(i,j=(5,3),v=1): - u_3_1 + u_5_1 + 4 x_(5,3)_1 <= 3
 subtours_3_(i,j=(5,4),v=1): - u_4_1 + u_5_1 + 4 x_(5,4)_1 <= 3
Bounds
Binaries
 y_1_1 x_(1,1)_1 x_(1,2)_1 x_(1,3)_1 x_(1,4)_1 x_(1,5)_1 y_2_1 x_(2,1)_1
 x_(2,2)_1 x_(2,3)_1 x_(2,4)_1 x_(2,5)_1 y_3_1 x_(3,1)_1 x_(3,2)_1
 x_(3,3)_1 x_(3,4)_1 x_(3,5)_1 y_4_1 x_(4,1)_1 x_(4,2)_1 x_(4,3)_1
 x_(4,4)_1 x_(4,5)_1 y_5_1 x_(5,1)_1 x_(5,2)_1 x_(5,3)_1 x_(5,4)_1
 x_(5,5)_1
Generals
 u_1_1 u_2_1 u_3_1 u_4_1 u_5_1
End
