#!/usr/bin/env node

/*
 * Copyright (C) 2023 Amazon.com, Inc. or its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

import { Construct } from 'constructs';
import { CacheStorage } from '@project-lakechain/core';
import { S3EventTrigger } from '@project-lakechain/s3-event-trigger';
import { SharpImageTransform, sharp } from '@project-lakechain/sharp-image-transform';
import { S3StorageConnector } from '@project-lakechain/s3-storage-connector';
import { RekognitionImageProcessor, dsl as r } from '@project-lakechain/rekognition-image-processor';
import { ImageLayerProcessor, dsl as l } from '@project-lakechain/image-layer-processor';
import { ControlNetImageProcessor } from '@project-lakechain/controlnet-image-processor';
import { Blip2ImageProcessor } from '@project-lakechain/blip2-image-processor';

/**
 * Example stack for running transformations on images
 * using the Sharp library.
 * The pipeline looks as follows:
 *
 * ┌────────────┐   ┌─────────────────────────┐   ┌─────────────┐
 * │  S3 Input  ├──►│  Sharp Image Processor  ├──►│  S3 Output  │
 * └────────────┘   └─────────────────────────┘   └─────────────┘
 *
 */
export class ImageTransformsStack extends cdk.Stack {

  /**
   * Stack constructor.
   */
  constructor(scope: Construct, id: string, env: cdk.StackProps) {
    super(scope, id, {
      description: 'A pipeline applying transformations on images.',
      ...env
    });

    // Sample VPC.
    const vpc = new ec2.Vpc(this, "VPC", {
      subnetConfiguration: [{
          subnetType: ec2.SubnetType.PUBLIC,
          name: 'Public',
        },{
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          name: 'Private',
        },{
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          name: 'Isolated',
        }
      ]
    });

    ///////////////////////////////////////////
    ///////         S3 Storage          ///////
    ///////////////////////////////////////////

    // The source bucket.
    const source = new s3.Bucket(this, 'Bucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      autoDeleteObjects: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      enforceSSL: true
    });

    // The destination bucket.
    const destination = new s3.Bucket(this, 'Destination', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      autoDeleteObjects: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      enforceSSL: true
    });

    // The cache storage.
    const cache = new CacheStorage(this, 'Cache', {});

    ///////////////////////////////////////////
    ///////     Lakechain Pipeline      ///////
    ///////////////////////////////////////////

    // Create the S3 trigger monitoring the bucket
    // for uploaded objects.
    const trigger = new S3EventTrigger.Builder()
      .withScope(this)
      .withIdentifier('Trigger')
      .withCacheStorage(cache)
      .withBucket(source)
      .build();


    // The Rekognition image processor will
    // identify objects in processed images.
    const rekognition = new RekognitionImageProcessor.Builder()
    .withScope(this)
    .withIdentifier('Rekognition')
    .withCacheStorage(cache)
    .withSource(trigger)
    .withIntent(
      r.detect()
        .labels(r.confidence(50))
    )
    .build();

    // Create a Sharp transform which will :
    // - Resize images to a width of 500px
    // - Grayscale images
    // - Flip images
    // - Convert images to PNG
    const imageTransform = new SharpImageTransform.Builder()
      .withScope(this)
      .withIdentifier('SharpTransform')
      .withCacheStorage(cache)
      .withSource(trigger)
      .withSharpTransforms(
        sharp()
          .resize(1024, 1024)
          .grayscale()
          .png()
      )
      .build();

      // Create the ControlNet image processor.
    const controlnetProcessor = new ControlNetImageProcessor.Builder()
      .withScope(this)
      .withIdentifier('ImageProcessor')
      .withCacheStorage(cache)
      .withVpc(vpc)
      .withSource(imageTransform) // 👈 Specify a data source
    .build();
    

    // Write the results to the destination bucket.
    new S3StorageConnector.Builder()
      .withScope(this)
      .withIdentifier('S3StorageConnector')
      .withCacheStorage(cache)
      .withDestinationBucket(destination)
      .withSources([imageTransform, controlnetProcessor, rekognition])
      .build();

    // Display the source bucket information in the console.
    new cdk.CfnOutput(this, 'SourceBucketName', {
      description: 'The name of the source bucket.',
      value: source.bucketName
    });

    // Display the destination bucket information in the console.
    new cdk.CfnOutput(this, 'DestinationBucketName', {
      description: 'The name of the destination bucket.',
      value: destination.bucketName
    });
  }
}

// Creating the CDK application.
const app = new cdk.App();

// Environment variables.
const account = process.env.CDK_DEFAULT_ACCOUNT ?? process.env.AWS_DEFAULT_ACCOUNT;
const region  = process.env.CDK_DEFAULT_REGION ?? process.env.AWS_DEFAULT_REGION;

// Deploy the stack.
new ImageTransformsStack(app, 'ImageTransformsStack', {
  env: {
    account,
    region
  }
});
