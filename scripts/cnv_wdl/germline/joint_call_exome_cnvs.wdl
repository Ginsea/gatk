version 1.0

workflow JointCallExomeCNVs {

    ##################################
    #### required basic arguments ####
    ##################################
    input {
      File intervals
      File? blacklist_intervals
      Array[File]+ segments_vcfs
      Array[File]+ segments_vcf_indexes
      Array[File]+ intervals_vcf
      Array[File]+ intervals_vcf_indexes
      Array[Array[File]] gcnv_calls_tars
      Array[File] gcnv_model_tars
      Array[File] calling_configs
      Array[File] denoising_configs
      Array[File] gcnvkernel_version
      Array[File] sharded_interval_lists
      File contig_ploidy_calls_tar
      Array[String]? allosomal_contigs
      Int ref_copy_number_autosomal_contigs
      File ref_fasta_dict
      File ref_fasta_fai
      File ref_fasta
      String gatk_docker
    }

    call JointSegmentation {
      input:
        segments_vcfs = segments_vcfs,
        segments_vcf_indexes = segments_vcf_indexes,
        ref_fasta = ref_fasta,
        ref_fasta_fai = ref_fasta_fai,
        ref_fasta_dict = ref_fasta_dict,
        gatk_docker = gatk_docker
    }

    Array[Array[File] ]gcnv_calls_tars_T = transpose(gcnv_calls_tars)

    scatter (scatter_index in range(length(segments_vcfs))) {
      call PostprocessGermlineCNVCalls as RecalcQual {
        input:
              entity_id = sub(sub(basename(intervals_vcf[scatter_index]), ".vcf.gz", ""), "intervals_output_", ""),
              gcnv_calls_tars = gcnv_calls_tars_T[scatter_index],
              gcnv_model_tars = gcnv_model_tars,
              calling_configs = calling_configs,
              denoising_configs = denoising_configs,
              gcnvkernel_version = gcnvkernel_version,
              sharded_interval_lists = sharded_interval_lists,
              contig_ploidy_calls_tar = contig_ploidy_calls_tar,
              allosomal_contigs = allosomal_contigs,
              ref_copy_number_autosomal_contigs = ref_copy_number_autosomal_contigs,
              sample_index = scatter_index,
              intervals_vcf = intervals_vcf[scatter_index],
              intervals_vcf_index = intervals_vcf_indexes[scatter_index],
              clustered_vcf = JointSegmentation.clustered_vcf,
              clustered_vcf_index = JointSegmentation.clustered_vcf_index,
              gatk_docker = gatk_docker
      }
    }

    call CombineVariants {
      input:
        input_vcfs = RecalcQual.genotyped_segments_vcf,
        ref_fasta = ref_fasta,
        ref_fasta_fai = ref_fasta_fai,
        ref_fasta_dict = ref_fasta_dict
    }
    output {
      File combined_calls = CombineVariants.combined_vcf
      File combined_calls_index = CombineVariants.combined_vcf_index
    }
}

task JointSegmentation {
  input {
    Array[File] segments_vcfs
    Array[File] segments_vcf_indexes
    File ref_fasta_dict
    File ref_fasta_fai
    File ref_fasta

     # Runtime parameters
    String gatk_docker
    Int? mem_gb
    Int? disk_space_gb
    Boolean use_ssd = false
    Int? cpu
    Int? preemptible_attempts
    }

    parameter_meta {
      segments_vcfs: {localization_optional: true}
      segments_vcf_indexes: {localization_optional: true}
    }

    Int machine_mem_mb = select_first([mem_gb, 2]) * 1000
    Int command_mem_mb = machine_mem_mb - 500

  #NOTE: output has to be gzipped to be read in by pyvcf in the next step
  command <<<
    set -e
    gatk --java-options "-Xmx~{command_mem_mb}m" JointCNVSegmentation \
    -R ~{ref_fasta} -O clustered.vcf.gz -V ~{sep=' -V ' segments_vcfs} --disable-sequence-dictionary-validation
    >>>

    output {
      File clustered_vcf = "clustered.vcf.gz"
      File clustered_vcf_index = "clustered.vcf.gz.tbi"
    }

    runtime {
      docker: gatk_docker
      memory: machine_mem_mb + " MB"
      disks: "local-disk " + select_first([disk_space_gb, 40]) + if use_ssd then " SSD" else " HDD"
      cpu: select_first([cpu, 1])
      preemptible: select_first([preemptible_attempts, 2])
    }
}

task CombineVariants {
  input {
    Array[File] input_vcfs
    File ref_fasta
    File ref_fasta_fai
    File ref_fasta_dict
    Int? preemptible_tries
    Int? disk_size
  }

  command <<<
    java -jar -XX:GCTimeLimit=50 -XX:GCHeapFreeLimit=10 -Xms2000m \
          -jar /usr/gitc/GATK35.jar \
          -T CombineVariants -R ~{ref_fasta} \
          -o combined.vcf
    >>>

  runtime {
    docker: "us.gcr.io/broad-gotc-prod/genomes-in-the-cloud:2.4.3-1564508330"
    preemptible: select_first([preemptible_tries, 2])
    memory: "3.5 GiB"
    cpu: "1"
    disks: "local-disk " + select_first([disk_size, 50]) + " HDD"
  }

  output {
    File combined_vcf = "combined.vcf"
    File combined_vcf_index = "combined.vcf.idx"
  }
}

#copied here instead of imported because common tasks are still on draft-3
task PostprocessGermlineCNVCalls {
    input {
      String entity_id
      Array[File] gcnv_calls_tars
      Array[File] gcnv_model_tars
      Array[File] calling_configs
      Array[File] denoising_configs
      Array[File] gcnvkernel_version
      Array[File] sharded_interval_lists
      File contig_ploidy_calls_tar
      Array[String]? allosomal_contigs
      Int ref_copy_number_autosomal_contigs
      Int sample_index
      File? intervals_vcf
      File? intervals_vcf_index
      File? clustered_vcf
      File? clustered_vcf_index
      File? gatk4_jar_override

      # Runtime parameters
      String gatk_docker
      Int? mem_gb
      Int? disk_space_gb
      Boolean use_ssd = false
      Int? cpu
      Int? preemptible_attempts
    }

    Int machine_mem_mb = select_first([mem_gb, 7]) * 1000
    Int command_mem_mb = machine_mem_mb - 1000

    String genotyped_intervals_vcf_filename = "genotyped-intervals-~{entity_id}.vcf.gz"
    String genotyped_segments_vcf_filename = "genotyped-segments-~{entity_id}.vcf.gz"
    String denoised_copy_ratios_filename = "denoised_copy_ratios-~{entity_id}.tsv"

    Array[String] allosomal_contigs_args = if defined(allosomal_contigs) then prefix("--allosomal-contig ", select_first([allosomal_contigs])) else []

    command <<<
        set -e
        #I can just build the docker, because I can't remember where the jar is
        #export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk4_jar_override}

        sharded_interval_lists_array=(~{sep=" " sharded_interval_lists})

        # untar calls to CALLS_0, CALLS_1, etc directories and build the command line
        # also copy over shard config and interval files
        gcnv_calls_tar_array=(~{sep=" " gcnv_calls_tars})
        calling_configs_array=(~{sep=" " calling_configs})
        denoising_configs_array=(~{sep=" " denoising_configs})
        gcnvkernel_version_array=(~{sep=" " gcnvkernel_version})
        sharded_interval_lists_array=(~{sep=" " sharded_interval_lists})
        calls_args=""
        for index in ${!gcnv_calls_tar_array[@]}; do
            gcnv_calls_tar=${gcnv_calls_tar_array[$index]}
            mkdir -p CALLS_$index/SAMPLE_~{sample_index}
            tar xzf $gcnv_calls_tar -C CALLS_$index/SAMPLE_~{sample_index}
            cp ${calling_configs_array[$index]} CALLS_$index/
            cp ${denoising_configs_array[$index]} CALLS_$index/
            cp ${gcnvkernel_version_array[$index]} CALLS_$index/
            cp ${sharded_interval_lists_array[$index]} CALLS_$index/
            calls_args="$calls_args --calls-shard-path CALLS_$index"
        done

        # untar models to MODEL_0, MODEL_1, etc directories and build the command line
        gcnv_model_tar_array=(~{sep=" " gcnv_model_tars})
        model_args=""
        for index in ${!gcnv_model_tar_array[@]}; do
            gcnv_model_tar=${gcnv_model_tar_array[$index]}
            mkdir MODEL_$index
            tar xzf $gcnv_model_tar -C MODEL_$index
            model_args="$model_args --model-shard-path MODEL_$index"
        done

        mkdir contig-ploidy-calls
        tar xzf ~{contig_ploidy_calls_tar} -C contig-ploidy-calls

        gatk --java-options "-Xmx~{command_mem_mb}m" PostprocessGermlineCNVCalls \
            $calls_args \
            $model_args \
            ~{sep=" " allosomal_contigs_args} \
            --autosomal-ref-copy-number ~{ref_copy_number_autosomal_contigs} \
            --contig-ploidy-calls contig-ploidy-calls \
            --sample-index ~{sample_index} \
            --output-genotyped-intervals ~{genotyped_intervals_vcf_filename} \
            --output-genotyped-segments ~{genotyped_segments_vcf_filename} \
            --output-denoised-copy-ratios ~{denoised_copy_ratios_filename} \
            ~{"--combined-intervals-vcf " + intervals_vcf} \
            ~{"--clustered-breakpoints " + clustered_vcf}

        rm -rf CALLS_*
        rm -rf MODEL_*
        rm -rf contig-ploidy-calls
    >>>

    runtime {
        docker: gatk_docker
        memory: machine_mem_mb + " MB"
        disks: "local-disk " + select_first([disk_space_gb, 40]) + if use_ssd then " SSD" else " HDD"
        cpu: select_first([cpu, 1])
        preemptible: select_first([preemptible_attempts, 5])
    }

    output {
        File genotyped_intervals_vcf = genotyped_intervals_vcf_filename
        File genotyped_segments_vcf = genotyped_segments_vcf_filename
        File denoised_copy_ratios = denoised_copy_ratios_filename
    }
}