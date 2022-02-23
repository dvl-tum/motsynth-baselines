from torchreid.engine import ImageSoftmaxEngine
import numpy as np

class ImageSoftmaxEngineSeveralSeq(ImageSoftmaxEngine):
    """We just modify torchreid's ImageSoftmaxEngine slightly so that it also reports
    avg mAP and rank-1 across all test datasets (MOTS video sequences in our case).
    """
    def test(
        self,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        r"""Tests model on target datasets.
        .. note::
            This function has been called in ``run()``.
        .. note::
            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())
        
        rank1s = []
        mAPs = []
        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']
            rank1, mAP = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )

            if self.writer is not None:
                self.writer.add_scalar(f'Test/{name}/rank1', rank1, self.epoch)
                self.writer.add_scalar(f'Test/{name}/mAP', mAP, self.epoch)
            
            rank1s.append(rank1)
            mAPs.append(mAP)

        avg_mAP = np.mean(np.array(mAPs))
        avg_rank1 = np.mean(np.array(rank1s))
        print('** OVERALL Results **')
        print('OVERALL mAP: {:.1%}'.format(avg_mAP))
        print('OVERALL Rank-1: {:.1%}'.format(avg_rank1))
        if self.writer is not None:
            self.writer.add_scalar(f'Test/OVERALL/rank1', avg_rank1, self.epoch)
            self.writer.add_scalar(f'Test/OVERALL/mAP', avg_mAP, self.epoch)

        return rank1
