# Graphviz 설치 (노드와 화살표 시각화)
digraph g { 
1 [label = "x" color = orange, style = filled]
2 [label = "y" color = orange, style = filed] 
3 [label = "Exp" color = lightblue, style = filled, shape = box]
}

digraph g { 
1 [label = "x" color = orange, style = filled]
2 [label = "y" color = orange, style = filed] 
3 [label = "Exp" color = lightblue, style = filled, shape = box]
1 -> 3
3 -> 2
}