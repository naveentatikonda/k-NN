/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;

import java.io.IOException;

@AllArgsConstructor
@Log4j2
public class LuceneEngineRescoreKnnVectorQuery extends Query {
    @Getter
    private final Query luceneQuery;
    private final int k;
    private final float[] queryVector;

    /*
      Prevents repeated rewrites of the query for the Lucene engine.
    */
    @Override
    public Query rewrite(IndexSearcher indexSearcher) {
        return this;
    }

    /*
       Rewrites the query just before weight creation.
     */
    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        Query rewrittenQuery = luceneQuery.rewrite(searcher);
        IndexReader reader = searcher.getIndexReader();
        Weight weight = searcher.createWeight(rewrittenQuery, ScoreMode.COMPLETE_NO_SCORES, 1.0f);
        KnnFloatVectorQuery query = (KnnFloatVectorQuery) luceneQuery;
        HitQueue queue = new HitQueue(query.getK(), false);
        for (var leaf : reader.leaves()) {
            Scorer scorer = weight.scorer(leaf);
            if (scorer == null) {
                continue;
            }
            FloatVectorValues floatVectorValues = leaf.reader().getFloatVectorValues(query.getField());
            if (floatVectorValues == null) {
                continue;
            }
            FieldInfo fi = leaf.reader().getFieldInfos().fieldInfo(query.getField());
            if (fi == null) {
                continue;
            }
            VectorSimilarityFunction comparer = fi.getVectorSimilarityFunction();
            DocIdSetIterator iterator = scorer.iterator();
            while (iterator.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                int docId = iterator.docID();
                // float[] vectorValue = floatVectorValues.vectorValue(docId);
                float[] vectorValue = floatVectorValues.vectorValue();
                float score = comparer.compare(vectorValue, query.getTargetCopy());
                queue.insertWithOverflow(new ScoreDoc(leaf.docBase + docId, score));
            }
        }
        int i = 0;
        ScoreDoc[] scoreDocs = new ScoreDoc[queue.size()];
        for (ScoreDoc topDoc : queue) {
            scoreDocs[i++] = topDoc;
        }
        // return createRewrittenQuery(reader, scoreDocs).createWeight(searcher, scoreMode, boost);
        return searcher.createWeight(rewrittenQuery, ScoreMode.COMPLETE_NO_SCORES, 1.0f);
    }

    @Override
    public String toString(String s) {
        return luceneQuery.toString();
    }

    @Override
    public void visit(QueryVisitor queryVisitor) {
        queryVisitor.visitLeaf(this);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LuceneEngineRescoreKnnVectorQuery otherQuery = (LuceneEngineRescoreKnnVectorQuery) o;
        return luceneQuery.equals(otherQuery.luceneQuery);
    }

    @Override
    public int hashCode() {
        return luceneQuery.hashCode();
    }
}
