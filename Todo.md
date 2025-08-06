# ToDo Lists

- [ ] Add support for AWS S3 for image upload
- [ ] Add support for AWS S3 vector storage
    - [Amazon S3 Vector Buckets Hands on – Similar Product Search Using: Image and Text Retrieval with RRF Fusion](https://medium.com/srcecde/similar-product-search-using-amazon-s3-vector-buckets-image-and-text-retrieval-with-rrf-fusion-adee7b484adf)
- [x] Add support for MUVERA for faster search
    - ColPali model is a multi-vector model, which generates embeddings per patches.
    - Vector search over all patch-embeddings is definitely a bad idea.
    - Use MUVERA to quantize the multi-vectors into fixed-sized single vector!
