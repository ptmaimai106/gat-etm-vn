# Tại Sao Hierarchy Quan Trọng Trong GAT-ETM?

## Tổng Quan

Knowledge Graph trong GAT-ETM được tổ chức dạng **hierarchical structure** (cấu trúc phân cấp) cho cả ICD và ATC codes. Tài liệu này giải thích **tại sao** cấu trúc này quan trọng và **ý nghĩa** của nó trong việc training GAT-ETM.

---

## 1. SEMANTIC STRUCTURE (Cấu Trúc Ngữ Nghĩa)

### Vấn đề:
- Medical codes không độc lập, chúng có mối quan hệ ngữ nghĩa với nhau
- Ví dụ: "4019" (hypertension) là một loại của "401" (circulatory diseases)

### Giải pháp - Hierarchy:
```
ICD9_ROOT (level 0)
  └── 401 (level 3) - "Diseases of circulatory system"
      └── 4019 (level 4) - "Essential hypertension, unspecified"
```

### Lợi ích:
- **Cấu trúc "is-a"**: Hierarchy thể hiện mối quan hệ "là một loại của"
- **Semantic relationships**: GAT hiểu được codes nào thuộc cùng một nhóm
- **Knowledge-based**: Dựa trên ontology y tế (ICD-9, ATC classification)

### Trong GAT-ETM:
- GAT sử dụng hierarchy để tính **attention weights**
- Attention mechanism tập trung vào các nodes liên quan trong hierarchy
- Model học được semantic similarity giữa các codes

---

## 2. GENERALIZATION (Tổng Quát Hóa)

### Vấn đề:
- **Rare codes**: Nhiều ICD/ATC codes hiếm gặp trong training data
- **Data sparsity**: Không đủ dữ liệu để học embeddings tốt cho mọi code
- **Cold start**: Codes mới chưa từng xuất hiện trong training

### Giải pháp - Hierarchy:
```
Parent node (401): Common, nhiều dữ liệu
  └── Child node (4019): Rare, ít dữ liệu
```

### Lợi ích:
- **Knowledge transfer**: Model học từ parent nodes (common) và áp dụng cho child nodes (rare)
- **Generalization**: Nếu biết "401" (parent), có thể suy luận về "4019" (child)
- **Better embeddings**: Rare codes vẫn có embeddings tốt nhờ kế thừa từ parent

### Ví dụ cụ thể:
```
Scenario: "4019" (hypertension) là rare code, chỉ xuất hiện 10 lần trong training

KHÔNG có hierarchy:
  - Model chỉ học từ 10 examples → embedding kém
  - Không thể generalize sang codes tương tự

CÓ hierarchy:
  - "401" (parent) xuất hiện 1000 lần → embedding tốt
  - "4019" kế thừa features từ "401" + specific features
  - Final embedding = f(parent_embedding, specific_features)
  - Kết quả: Embedding tốt hơn dù chỉ có 10 examples
```

---

## 3. KNOWLEDGE TRANSFER (Chuyển Giao Tri Thức)

### Vấn đề:
- Codes cùng nhóm có semantic similarity nhưng không có direct connection
- Ví dụ: "4019" và "4011" đều là circulatory diseases nhưng không cùng xuất hiện

### Giải pháp - Hierarchy:
```
401 (parent)
  ├── 4019 (child 1)
  └── 4011 (child 2)
```

### Lợi ích:
- **Semantic similarity**: Codes cùng parent có embeddings tương tự
- **Pattern sharing**: Nếu "4019" liên quan đến một thuốc, "4011" cũng có thể liên quan
- **Few-shot learning**: Học được patterns từ ít dữ liệu hơn

### Trong GAT-ETM:
- GAT attention mechanism: Codes cùng parent có attention weights cao hơn
- Embedding learning: Child nodes kế thừa features từ parent
- Topic modeling: Topics được học từ cả parent và child nodes

---

## 4. GRAPH ATTENTION NETWORK (GAT) MECHANISM

### Cách GAT sử dụng Hierarchy:

#### **Layer 1**: Local neighbors
```
Node "4019" attends to:
  - Direct neighbors (co-occurrence edges)
  - Parent "401" (hierarchical edge)
```

#### **Layer 2**: Extended neighbors
```
Node "4019" attends to:
  - Neighbors of "401" (grandparent, siblings)
  - 2-hop neighbors via hierarchy
```

#### **Layer 3**: Global context
```
Node "4019" attends to:
  - All ancestors (ICD9_ROOT, 401)
  - All descendants (nếu có)
  - Related codes qua hierarchy
```

### Code Implementation:
```python
# graph_etm.py - GCNet class
class GCNet(nn.Module):
    def __init__(self, num_nodes, num_feature, node_embeddings=None):
        self.num_layers = 3
        self.gcns = torch.nn.ModuleList()
        for i in range(self.num_layers):
            # GATConv uses hierarchy edges to compute attention
            self.gcns.append(gnn.GATConv(num_feature, num_feature, heads=4))
    
    def forward(self, data):
        x = self.init.weight  # Initial embeddings
        embed_rep = [x]
        
        for i in range(self.num_layers):
            # Each layer learns from hierarchy
            x = self.gcns[i](x, edge_index)  # edge_index includes hierarchical edges
            embed_rep.append(x)
        
        # Combine all layers → final embedding ρ
        output = self.fc(torch.cat(embed_rep, dim=1))
        return output  # Learned code embedding ρ
```

### Attention Weights:
- **Hierarchical edges**: High attention (semantic relationship)
- **Co-occurrence edges**: Medium attention (empirical pattern)
- **Augmented edges**: Weighted by distance (0.9^distance)

---

## 5. EMBEDDING LEARNING

### Hierarchy giúp học embeddings tốt hơn:

#### **Parent Nodes**:
- Đại diện cho **general concepts**
- Embeddings capture **common features** của toàn bộ subtree
- Ví dụ: "401" embedding = average of all circulatory diseases

#### **Child Nodes**:
- Kế thừa features từ parent
- Thêm **specific features** của riêng mình
- Final embedding = `f(parent_embedding, specific_features)`

### Ví dụ:
```
"401" (parent) embedding:
  - General: [circulatory, cardiovascular, blood_pressure, ...]
  - Learned from: 1000+ examples

"4019" (child) embedding:
  - Inherited: [circulatory, cardiovascular, blood_pressure, ...]
  - Specific: [hypertension, essential, unspecified, ...]
  - Learned from: 10 examples + parent knowledge
```

---

## 6. AUGMENTATION (Skip Connections)

### Tại sao cần augmentation?

#### **Vấn đề với chỉ có direct hierarchy**:
```
ICD9_ROOT → 401 → 4019
```
- Information flow chậm (phải đi qua nhiều layers)
- Long-range dependencies khó học

#### **Giải pháp - Augmentation**:
```
ICD9_ROOT → 401 → 4019
  └──────────────┘ (skip connection)
```
- Kết nối mỗi node với **tất cả ancestors**
- Weight decay: `0.9^distance` (càng xa càng nhẹ)
- Giúp information flow nhanh hơn

### Code Implementation:
```python
# build_kg_paper_simple.py - augment_graph()
for node in G.nodes():
    ancestors = find_all_ancestors(node)  # Via hierarchical edges
    for ancestor in ancestors:
        distance = abs(node.level - ancestor.level)
        weight = 0.9 ** distance
        G.add_edge(node, ancestor, 
                   edge_type='augmented', 
                   weight=weight)
```

### Lợi ích:
- **Faster information flow**: Direct connections to ancestors
- **Long-range dependencies**: Model học được relationships xa trong hierarchy
- **Better gradients**: Backpropagation dễ dàng hơn

---

## 7. SO SÁNH: CÓ vs KHÔNG CÓ HIERARCHY

### **KHÔNG có Hierarchy**:
```
❌ Codes độc lập, không có semantic structure
❌ Rare codes có embeddings kém
❌ Không thể generalize sang codes mới
❌ Model chỉ học từ co-occurrence (data-driven)
❌ Thiếu knowledge-based information
```

### **CÓ Hierarchy**:
```
✅ Codes có semantic structure (is-a relationships)
✅ Rare codes có embeddings tốt nhờ parent knowledge
✅ Có thể generalize sang codes mới
✅ Model học từ cả hierarchy (knowledge-based) và co-occurrence (data-driven)
✅ Kết hợp được ontology knowledge và empirical patterns
```

---

## 8. KẾT LUẬN

### Hierarchy quan trọng vì:

1. **Semantic Structure**: Cung cấp cấu trúc ngữ nghĩa (is-a relationships)
2. **Generalization**: Giúp model tổng quát hóa từ parent sang child
3. **Knowledge Transfer**: Codes cùng parent chia sẻ knowledge
4. **GAT Mechanism**: GAT sử dụng hierarchy để tính attention weights
5. **Embedding Learning**: Embeddings tốt hơn cho cả common và rare codes
6. **Augmentation**: Skip connections giúp information flow nhanh hơn

### Trong GAT-ETM:
- **Input**: Knowledge Graph với hierarchical structure
- **Process**: GAT learns embeddings từ hierarchy + co-occurrence
- **Output**: Code embeddings ρ được sử dụng trong ETM topic model
- **Result**: Model hiểu được cả semantic structure và empirical patterns

### Best Practice:
- ✅ **Luôn build hierarchy** cho ICD và ATC codes
- ✅ **Augment graph** với skip connections
- ✅ **Kết hợp** hierarchical edges và co-occurrence edges
- ✅ **Sử dụng** cả knowledge-based và data-driven information

---

## Tài Liệu Tham Khảo

- GAT-ETM Paper: "Modeling electronic health record data using an end-to-end knowledge-graph-informed topic model"
- Graph Attention Networks (GAT): Velickovic et al., 2018
- ICD-9 Hierarchy: https://www.cdc.gov/nchs/icd/icd9cm.htm
- ATC Classification: https://www.whocc.no/atc_ddd_index/
