# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# PDF parser logic
import os
from typing import Dict, Any
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal

class PDFParser:
    """Parser for PDF documents"""

    def visualize_clusters(self, X, labels):
        """Visualize the clusters using a scatter plot."""
        plt.figure(figsize=(10, 8))
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'k'  # Black used for noise.

            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

        plt.title('DBSCAN Clustering of Text Blocks')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

    def extract_text_with_clustering(self, pdf_path):
        text_blocks = []
        for page_layout in extract_pages(pdf_path):
            for element in page_layout:
                if isinstance(element, LTTextBoxHorizontal):
                    x0, y0, x1, y1 = element.bbox
                    text = element.get_text()
                    text_blocks.append((x0, y0, x1, y1, len(text), text))

        # Convert to numpy array for clustering
        X = np.array([(x0, y0, x1, y1, len(text)) for x0, y0, x1, y1, _, text in text_blocks])
        clustering = DBSCAN(eps=10, min_samples=2).fit(X)

        # Identify the cluster with the most points as the body text
        labels, counts = np.unique(clustering.labels_, return_counts=True)
        body_text_cluster = labels[np.argmax(counts)]

        # Extract text from the identified body text cluster
        extracted_text = [text for (_, _, _, _, _, text), label in zip(text_blocks, clustering.labels_) if label == body_text_cluster]

        # Visualize the clusters
        # self.visualize_clusters(X, clustering.labels_)

        return '\n'.join(extracted_text)
    
    def parse(self, file_path: str) -> str:
        """Parse a PDF file into plain text
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        try:
            # from pdfminer.high_level import extract_text
            # return extract_text(file_path)
            return self.extract_text_with_clustering(file_path)
        except ImportError:
            raise ImportError("pdfminer.six is required for PDF parsing. Install it with: pip install pdfminer.six")
    
    def save(self, content: str, output_path: str) -> None:
        """Save the extracted text to a file
        
        Args:
            content: Extracted text content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)