## Decoder Performance Comparison  

1. **MLD (Maximum Likelihood Decoder)**  
   - Limited to:  
     - r=1 cases  
     - d=3, r=3 surface codes  
   - Both accuracy and speed benchmarks were performed only for these configurations  

2. **EMLD (Efficient Maximum Likelihood Decoder)**  
   - Capable of handling:  
     - All r=1 cases  
     - All d=3, r=3 surface codes  
   - Demonstrates broader applicability than MLD  

3. **EAMLD (Efficient Approximate Maximum Likelihood Decoder)**  
   - Successfully processes all tested configurations:  
     - All surface code cases (varying r and d)  
     - All QLDPC code cases  
   - Shows complete coverage of the experimental parameter space  

4. **BP+OSD (Belief Propagation with Ordered Statistics Decoding)**  
   - **QLDPC-specific performance**:  
     - Successfully handles all r=1 cases  
     - For r=d configurations, exhibits significantly slower decoding speed  
   - Remains a competitive baseline for certain QLDPC scenarios  

5. **EAPMLD (Efficient Approximate Parallel Maximum Likelihood Decoder)**  
   - Processes all configurations successfully  
   - Current implementation shows approximately 3× slower performance compared to other methods  
   - Note: Potential speed advantages not yet realized in current version  

## Performance Characteristics  
The results demonstrate a clear progression in capability across decoders:  
- MLD shows fundamental limitations in code size support  
- EMLD expands to full r=1 and basic surface code coverage  
- EAMLD achieves universal coverage for both surface and QLDPC codes  
- BP+OSD provides QLDPC-specific decoding but with scaling challenges  
- EAPMLD offers complete coverage with future optimization potential  

This comprehensive evaluation highlights EAMLD's unique position as the only decoder successfully handling all tested code families while maintaining practical performance.