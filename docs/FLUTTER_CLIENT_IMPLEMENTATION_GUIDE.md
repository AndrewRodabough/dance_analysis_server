# Flutter Client Implementation Guide: Routines, Invites & Routine Viewing

**Target Audience**: Flutter developers implementing group collaboration features on the dance analysis client.

**Scope**: Routine management, group invitations, and routine viewing workflows.

---

## Table of Contents

1. [Data Models](#data-models)
2. [API Service Layer](#api-service-layer)
3. [Navigation Architecture](#navigation-architecture)
4. [State Management Patterns](#state-management-patterns)
5. [Screen/Page Structure](#screenpage-structure)
6. [End-to-End Flow Examples](#end-to-end-flow-examples)
7. [Error Handling & UX](#error-handling--ux)
8. [Testing Strategy](#testing-strategy)
9. [Implementation Checklist](#implementation-checklist)

---

## Data Models

Define these models in your `lib/models/` directory. These mirror the backend API responses.

### Groups & Members

```dart
// lib/models/group.dart

class Group {
  final String id;
  final String name;
  final String description;
  final DateTime createdAt;
  final bool isOwner; // Is the current user an owner of this group?

  Group({
    required this.id,
    required this.name,
    required this.description,
    required this.createdAt,
    required this.isOwner,
  });

  factory Group.fromJson(Map<String, dynamic> json) {
    return Group(
      id: json['id'] as String,
      name: json['name'] as String,
      description: json['description'] as String,
      createdAt: DateTime.parse(json['created_at'] as String),
      isOwner: json['is_owner'] as bool? ?? false,
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'description': description,
    'created_at': createdAt.toIso8601String(),
    'is_owner': isOwner,
  };
}

class GroupMember {
  final String userId;
  final String email;
  final String? displayName;
  final String role; // 'owner', 'member', etc.
  final DateTime joinedAt;

  GroupMember({
    required this.userId,
    required this.email,
    this.displayName,
    required this.role,
    required this.joinedAt,
  });

  factory GroupMember.fromJson(Map<String, dynamic> json) {
    return GroupMember(
      userId: json['user_id'] as String,
      email: json['email'] as String,
      displayName: json['display_name'] as String?,
      role: json['role'] as String? ?? 'member',
      joinedAt: DateTime.parse(json['joined_at'] as String),
    );
  }
}
```

### Group Invites

```dart
// lib/models/group_invite.dart

enum InviteStatus {
  pending,
  accepted,
  expired,
  declined;

  static InviteStatus fromString(String value) {
    return InviteStatus.values.firstWhere(
      (e) => e.name == value.toLowerCase(),
      orElse: () => InviteStatus.pending,
    );
  }
}

class GroupInvite {
  final String id;
  final String groupId;
  final String groupName;
  final String email;
  final String token; // Temporary: shown in UI until email service exists
  final InviteStatus status;
  final DateTime createdAt;
  final DateTime? expiresAt;
  final String? role;

  GroupInvite({
    required this.id,
    required this.groupId,
    required this.groupName,
    required this.email,
    required this.token,
    required this.status,
    required this.createdAt,
    this.expiresAt,
    this.role,
  });

  factory GroupInvite.fromJson(Map<String, dynamic> json) {
    return GroupInvite(
      id: json['id'] as String,
      groupId: json['group_id'] as String,
      groupName: json['group_name'] as String? ?? '',
      email: json['email'] as String,
      token: json['token'] as String? ?? '',
      status: InviteStatus.fromString(json['status'] as String? ?? 'pending'),
      createdAt: DateTime.parse(json['created_at'] as String),
      expiresAt: json['expires_at'] != null 
        ? DateTime.parse(json['expires_at'] as String)
        : null,
      role: json['role'] as String?,
    );
  }

  bool get isExpired => expiresAt != null && expiresAt!.isBefore(DateTime.now());
}
```

### Routines

```dart
// lib/models/routine.dart

class Routine {
  final String id;
  final String groupId;
  final String name;
  final String description;
  final DateTime createdAt;
  final DateTime? updatedAt;
  final int videoCount; // Cached count for display

  Routine({
    required this.id,
    required this.groupId,
    required this.name,
    required this.description,
    required this.createdAt,
    this.updatedAt,
    this.videoCount = 0,
  });

  factory Routine.fromJson(Map<String, dynamic> json) {
    return Routine(
      id: json['id'] as String,
      groupId: json['group_id'] as String,
      name: json['name'] as String,
      description: json['description'] as String,
      createdAt: DateTime.parse(json['created_at'] as String),
      updatedAt: json['updated_at'] != null
        ? DateTime.parse(json['updated_at'] as String)
        : null,
      videoCount: json['video_count'] as int? ?? 0,
    );
  }

  Map<String, dynamic> toJson() => {
    'name': name,
    'description': description,
  };
}
```

### Videos & Notes

```dart
// lib/models/routine_video.dart

enum VideoStatus {
  pendingUpload,
  uploaded,
  deleted;

  static VideoStatus fromString(String value) {
    const map = {
      'pending_upload': VideoStatus.pendingUpload,
      'uploaded': VideoStatus.uploaded,
      'deleted': VideoStatus.deleted,
    };
    return map[value] ?? VideoStatus.pendingUpload;
  }

  String toBackendString() {
    const map = {
      VideoStatus.pendingUpload: 'pending_upload',
      VideoStatus.uploaded: 'uploaded',
      VideoStatus.deleted: 'deleted',
    };
    return map[this] ?? 'pending_upload';
  }
}

class RoutineVideo {
  final String id;
  final String routineId;
  final String uploadedByUserId;
  final String uploadedByEmail;
  final VideoStatus status;
  final DateTime createdAt;
  final DateTime? uploadedAt;
  final String? fileSize; // Human-readable size

  RoutineVideo({
    required this.id,
    required this.routineId,
    required this.uploadedByUserId,
    required this.uploadedByEmail,
    required this.status,
    required this.createdAt,
    this.uploadedAt,
    this.fileSize,
  });

  factory RoutineVideo.fromJson(Map<String, dynamic> json) {
    return RoutineVideo(
      id: json['id'] as String,
      routineId: json['routine_id'] as String,
      uploadedByUserId: json['uploaded_by_user_id'] as String,
      uploadedByEmail: json['uploaded_by_email'] as String,
      status: VideoStatus.fromString(json['status'] as String),
      createdAt: DateTime.parse(json['created_at'] as String),
      uploadedAt: json['uploaded_at'] != null
        ? DateTime.parse(json['uploaded_at'] as String)
        : null,
      fileSize: json['file_size'] as String?,
    );
  }

  bool get isUploaded => status == VideoStatus.uploaded;
  bool get isPendingUpload => status == VideoStatus.pendingUpload;
}

// lib/models/routine_note.dart

class RoutineNote {
  final String id;
  final String routineId;
  final String? videoId; // Null if routine-level note
  final int? videoTimestampMs; // Null if routine-level note
  final String createdByEmail;
  final String content;
  final Map<String, dynamic>? details;
  final bool videoDeleted; // True if associated video was deleted
  final DateTime createdAt;

  RoutineNote({
    required this.id,
    required this.routineId,
    this.videoId,
    this.videoTimestampMs,
    required this.createdByEmail,
    required this.content,
    this.details,
    this.videoDeleted = false,
    required this.createdAt,
  });

  factory RoutineNote.fromJson(Map<String, dynamic> json) {
    return RoutineNote(
      id: json['id'] as String,
      routineId: json['routine_id'] as String,
      videoId: json['video_id'] as String?,
      videoTimestampMs: json['video_timestamp_ms'] as int?,
      createdByEmail: json['created_by_email'] as String,
      content: json['content'] as String,
      details: json['details'] as Map<String, dynamic>?,
      videoDeleted: json['video_deleted'] as bool? ?? false,
      createdAt: DateTime.parse(json['created_at'] as String),
    );
  }

  Map<String, dynamic> toJson() => {
    'content': content,
    'video_timestamp_ms': videoTimestampMs,
    'details': details,
  };

  bool get isVideoNote => videoId != null;
  String get formattedTimestamp {
    if (videoTimestampMs == null) return '';
    final seconds = videoTimestampMs! ~/ 1000;
    final minutes = seconds ~/ 60;
    final remainingSeconds = seconds % 60;
    return '${minutes}:${remainingSeconds.toString().padLeft(2, '0')}';
  }
}
```

### Upload Session (Local State)

```dart
// lib/models/upload_session.dart

class UploadSession {
  final String videoId;
  final String routineId;
  final String groupId;
  final String presignedUrl;
  final DateTime expiresAt;
  final File videoFile;
  double uploadProgress = 0.0; // 0.0 to 1.0

  UploadSession({
    required this.videoId,
    required this.routineId,
    required this.groupId,
    required this.presignedUrl,
    required this.expiresAt,
    required this.videoFile,
  });

  bool get isExpired => DateTime.now().isAfter(expiresAt);
  bool get isExpiringSoon => 
    DateTime.now().add(Duration(minutes: 5)).isAfter(expiresAt);
}
```

---

## API Service Layer

Create a service layer that encapsulates all backend communication.

```dart
// lib/services/api_client.dart

import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/group.dart';
import '../models/group_invite.dart';
import '../models/routine.dart';
import '../models/routine_video.dart';
import '../models/routine_note.dart';

class ApiClient {
  final String baseUrl;
  final String? authToken;
  
  const ApiClient({
    required this.baseUrl,
    this.authToken,
  });

  // ============== GROUPS ==============

  /// Fetch all groups for current user
  Future<List<Group>> getGroups() async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/groups'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map;
      final items = data['items'] as List? ?? [];
      return items.map((e) => Group.fromJson(e as Map<String, dynamic>)).toList();
    } else if (response.statusCode == 401) {
      throw UnauthorizedException('Session expired');
    } else {
      throw ApiException('Failed to fetch groups: ${response.statusCode}');
    }
  }

  /// Fetch a specific group with member list
  Future<GroupDetail> getGroupWithMembers(String groupId) async {
    final groupResponse = await http.get(
      Uri.parse('$baseUrl/api/v1/groups/$groupId'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (groupResponse.statusCode == 404) {
      throw NotFoundException('Group not found or no access');
    } else if (groupResponse.statusCode != 200) {
      throw ApiException('Failed to fetch group: ${groupResponse.statusCode}');
    }

    final group = Group.fromJson(
      jsonDecode(groupResponse.body) as Map<String, dynamic>
    );

    final membersResponse = await http.get(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/members'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    List<GroupMember> members = [];
    if (membersResponse.statusCode == 200) {
      final data = jsonDecode(membersResponse.body) as Map;
      final items = data['items'] as List? ?? [];
      members = items.map((e) => GroupMember.fromJson(e as Map<String, dynamic>)).toList();
    }

    return GroupDetail(group: group, members: members);
  }

  /// Create a new group
  Future<Group> createGroup({
    required String name,
    required String description,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/groups'),
      headers: _jsonHeaders(),
      body: jsonEncode({
        'name': name,
        'description': description,
      }),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 201 || response.statusCode == 200) {
      return Group.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    } else if (response.statusCode == 401) {
      throw UnauthorizedException('Not authenticated');
    } else {
      throw ApiException('Failed to create group: ${response.statusCode}');
    }
  }

  // ============== GROUP INVITES ==============

  /// Create and send a group invite
  Future<GroupInvite> createGroupInvite({
    required String groupId,
    required String email,
    String? role,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/invites'),
      headers: _jsonHeaders(),
      body: jsonEncode({
        'email': email.toLowerCase().trim(),
        if (role != null) 'role': role,
      }),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 201 || response.statusCode == 200) {
      return GroupInvite.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    } else if (response.statusCode == 404) {
      throw NotFoundException('Group not found or no access');
    } else if (response.statusCode == 400) {
      final error = jsonDecode(response.body) as Map;
      throw ApiException(error['detail'] as String? ?? 'Invalid invite');
    } else {
      throw ApiException('Failed to create invite: ${response.statusCode}');
    }
  }

  /// Get pending invites for current user
  Future<List<GroupInvite>> getPendingInvites() async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/group-invites/pending'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map;
      final items = data['items'] as List? ?? [];
      return items.map((e) => GroupInvite.fromJson(e as Map<String, dynamic>)).toList();
    } else {
      return [];
    }
  }

  /// Accept a group invite
  Future<Group> acceptGroupInvite(String token) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/group-invites/accept'),
      headers: _jsonHeaders(),
      body: jsonEncode({'token': token}),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      return Group.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    } else if (response.statusCode == 400 || response.statusCode == 404) {
      throw InvalidInviteException('Invite is invalid or expired.');
    } else {
      throw ApiException('Failed to accept invite: ${response.statusCode}');
    }
  }

  // ============== ROUTINES ==============

  /// Fetch all routines for a group
  Future<List<Routine>> getRoutines(String groupId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map;
      final items = data['items'] as List? ?? [];
      return items.map((e) => Routine.fromJson(e as Map<String, dynamic>)).toList();
    } else if (response.statusCode == 404) {
      throw NotFoundException('Group not found or no access');
    } else {
      throw ApiException('Failed to fetch routines: ${response.statusCode}');
    }
  }

  /// Fetch a specific routine
  Future<Routine> getRoutine(String groupId, String routineId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      return Routine.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    } else if (response.statusCode == 404) {
      throw NotFoundException('Routine not found or no access');
    } else {
      throw ApiException('Failed to fetch routine: ${response.statusCode}');
    }
  }

  /// Create a new routine in a group
  Future<Routine> createRoutine({
    required String groupId,
    required String name,
    required String description,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines'),
      headers: _jsonHeaders(),
      body: jsonEncode({
        'name': name,
        'description': description,
      }),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 201 || response.statusCode == 200) {
      return Routine.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    } else if (response.statusCode == 404) {
      throw NotFoundException('Group not found or no access');
    } else {
      throw ApiException('Failed to create routine: ${response.statusCode}');
    }
  }

  // ============== ROUTINE VIDEOS ==============

  /// Register a video upload and get presigned URL
  Future<VideoUploadRegistration> registerVideoUpload({
    required String groupId,
    required String routineId,
    required String fileName,
    required String fileSize,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/videos'),
      headers: _jsonHeaders(),
      body: jsonEncode({
        'file_name': fileName,
        'file_size': fileSize,
      }),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 201 || response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final video = RoutineVideo.fromJson(data['video'] as Map<String, dynamic>);
      final presignedUrl = data['upload_url'] as String;
      final expiresAt = DateTime.parse(data['expires_at'] as String);
      return VideoUploadRegistration(
        video: video,
        presignedUrl: presignedUrl,
        expiresAt: expiresAt,
      );
    } else if (response.statusCode == 404) {
      throw NotFoundException('Group or routine not found or no access');
    } else {
      throw ApiException('Failed to register upload: ${response.statusCode}');
    }
  }

  /// Upload video to presigned URL
  Future<void> uploadVideoFile({
    required String presignedUrl,
    required File file,
    required Function(double progress) onProgress,
  }) async {
    final fileLength = await file.length();
    final fileStream = file.openRead();
    
    final request = http.StreamedRequest('PUT', Uri.parse(presignedUrl))
      ..headers.addAll({
        'Content-Type': 'video/mp4', // Adjust based on file type
      })
      ..contentLength = fileLength;

    var uploaded = 0;
    fileStream.listen(
      (data) {
        request.sink.add(data);
        uploaded += data.length;
        onProgress(uploaded / fileLength);
      },
      onDone: () => request.sink.close(),
      onError: (error) => request.sink.close(),
    );

    final response = await request.send().timeout(const Duration(minutes: 10));
    if (response.statusCode != 200) {
      throw ApiException('Upload failed: ${response.statusCode}');
    }
  }

  /// Finalize a video upload
  Future<RoutineVideo> finalizeVideoUpload({
    required String groupId,
    required String routineId,
    required String videoId,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/videos/$videoId/finalize'),
      headers: _jsonHeaders(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      return RoutineVideo.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    } else if (response.statusCode == 404) {
      throw NotFoundException('Video not found or no access');
    } else {
      throw ApiException('Failed to finalize upload: ${response.statusCode}');
    }
  }

  /// Fetch videos for a routine
  Future<List<RoutineVideo>> getVideos({
    required String groupId,
    required String routineId,
    bool includeDeleted = false,
  }) async {
    final params = {
      if (includeDeleted) 'include_deleted': 'true',
    };

    final uri = Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/videos')
      .replace(queryParameters: params);

    final response = await http.get(
      uri,
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map;
      final items = data['items'] as List? ?? [];
      return items.map((e) => RoutineVideo.fromJson(e as Map<String, dynamic>)).toList();
    } else if (response.statusCode == 404) {
      throw NotFoundException('Group or routine not found or no access');
    } else {
      throw ApiException('Failed to fetch videos: ${response.statusCode}');
    }
  }

  /// Download a video (get presigned GET URL)
  Future<String> getVideoDownloadUrl({
    required String groupId,
    required String routineId,
    required String videoId,
  }) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/videos/$videoId/download'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map;
      return data['url'] as String;
    } else if (response.statusCode == 404) {
      throw NotFoundException('Video not found or no access');
    } else {
      throw ApiException('Failed to get download URL: ${response.statusCode}');
    }
  }

  /// Soft-delete a video
  Future<void> deleteVideo({
    required String groupId,
    required String routineId,
    required String videoId,
  }) async {
    final response = await http.delete(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/videos/$videoId'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 204) {
      // Success
    } else if (response.statusCode == 404) {
      throw NotFoundException('Video not found or no access');
    } else {
      throw ApiException('Failed to delete video: ${response.statusCode}');
    }
  }

  // ============== ROUTINE NOTES ==============

  /// Get routine-level notes
  Future<List<RoutineNote>> getRoutineNotes({
    required String groupId,
    required String routineId,
  }) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/notes'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map;
      final items = data['items'] as List? ?? [];
      return items.map((e) => RoutineNote.fromJson(e as Map<String, dynamic>)).toList();
    } else if (response.statusCode == 404) {
      throw NotFoundException('Routine not found or no access');
    } else {
      throw ApiException('Failed to fetch notes: ${response.statusCode}');
    }
  }

  /// Add a routine-level note
  Future<RoutineNote> addRoutineNote({
    required String groupId,
    required String routineId,
    required String content,
    Map<String, dynamic>? details,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/notes'),
      headers: _jsonHeaders(),
      body: jsonEncode({
        'content': content,
        if (details != null) 'details': details,
      }),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 201 || response.statusCode == 200) {
      return RoutineNote.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    } else if (response.statusCode == 404) {
      throw NotFoundException('Routine not found or no access');
    } else {
      throw ApiException('Failed to add note: ${response.statusCode}');
    }
  }

  /// Get video-specific notes
  Future<List<RoutineNote>> getVideoNotes({
    required String groupId,
    required String routineId,
    required String videoId,
  }) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/videos/$videoId/notes'),
      headers: _headers(),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map;
      final items = data['items'] as List? ?? [];
      return items.map((e) => RoutineNote.fromJson(e as Map<String, dynamic>)).toList();
    } else if (response.statusCode == 404) {
      throw NotFoundException('Video not found or no access');
    } else {
      throw ApiException('Failed to fetch video notes: ${response.statusCode}');
    }
  }

  /// Add a video-specific note with timestamp
  Future<RoutineNote> addVideoNote({
    required String groupId,
    required String routineId,
    required String videoId,
    required String content,
    int? videoTimestampMs,
    Map<String, dynamic>? details,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/groups/$groupId/routines/$routineId/videos/$videoId/notes'),
      headers: _jsonHeaders(),
      body: jsonEncode({
        'content': content,
        if (videoTimestampMs != null) 'video_timestamp_ms': videoTimestampMs,
        if (details != null) 'details': details,
      }),
    ).timeout(const Duration(seconds: 10));

    if (response.statusCode == 201 || response.statusCode == 200) {
      return RoutineNote.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    } else if (response.statusCode == 404) {
      throw NotFoundException('Video not found or no access');
    } else {
      throw ApiException('Failed to add note: ${response.statusCode}');
    }
  }

  // ============== HELPERS ==============

  Map<String, String> _headers() => {
    'Authorization': 'Bearer $authToken',
    'Accept': 'application/json',
  };

  Map<String, String> _jsonHeaders() => {
    ..._headers(),
    'Content-Type': 'application/json',
  };
}

// ============== DATA TRANSFER OBJECTS ==============

class GroupDetail {
  final Group group;
  final List<GroupMember> members;

  GroupDetail({required this.group, required this.members});
}

class VideoUploadRegistration {
  final RoutineVideo video;
  final String presignedUrl;
  final DateTime expiresAt;

  VideoUploadRegistration({
    required this.video,
    required this.presignedUrl,
    required this.expiresAt,
  });
}

// ============== EXCEPTIONS ==============

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  
  @override
  String toString() => 'ApiException: $message';
}

class UnauthorizedException extends ApiException {
  UnauthorizedException(String message) : super(message);
}

class NotFoundException extends ApiException {
  NotFoundException(String message) : super(message);
}

class InvalidInviteException extends ApiException {
  InvalidInviteException(String message) : super(message);
}
```

---

## Navigation Architecture

Set up routes for the new features. Adapt the example to your current navigation package.

```dart
// lib/routes/app_router.dart
// Example using GoRouter - adjust for your nav framework

import 'package:go_router/go_router.dart';
import '../screens/groups/groups_list_screen.dart';
import '../screens/groups/group_detail_screen.dart';
import '../screens/groups/create_group_screen.dart';
import '../screens/groups/manage_invites_screen.dart';
import '../screens/groups/invite_accept_screen.dart';
import '../screens/routines/routines_list_screen.dart';
import '../screens/routines/routine_detail_screen.dart';
import '../screens/routines/create_routine_screen.dart';
import '../screens/routines/video_upload_screen.dart';
import '../screens/routines/video_player_screen.dart';

final appRouter = GoRouter(
  routes: [
    GoRoute(
      path: '/groups',
      name: 'groups',
      builder: (context, state) => const GroupsListScreen(),
      routes: [
        GoRoute(
          path: 'create',
          name: 'createGroup',
          builder: (context, state) => const CreateGroupScreen(),
        ),
        GoRoute(
          path: ':groupId',
          name: 'groupDetail',
          builder: (context, state) => GroupDetailScreen(
            groupId: state.pathParameters['groupId']!,
          ),
          routes: [
            GoRoute(
              path: 'invites',
              name: 'manageInvites',
              builder: (context, state) => ManageInvitesScreen(
                groupId: state.pathParameters['groupId']!,
              ),
            ),
            GoRoute(
              path: 'routines',
              name: 'routines',
              builder: (context, state) => RoutinesListScreen(
                groupId: state.pathParameters['groupId']!,
              ),
              routes: [
                GoRoute(
                  path: 'create',
                  name: 'createRoutine',
                  builder: (context, state) => CreateRoutineScreen(
                    groupId: state.pathParameters['groupId']!,
                  ),
                ),
                GoRoute(
                  path: ':routineId',
                  name: 'routineDetail',
                  builder: (context, state) => RoutineDetailScreen(
                    groupId: state.pathParameters['groupId']!,
                    routineId: state.pathParameters['routineId']!,
                  ),
                  routes: [
                    GoRoute(
                      path: 'upload',
                      name: 'uploadVideo',
                      builder: (context, state) => VideoUploadScreen(
                        groupId: state.pathParameters['groupId']!,
                        routineId: state.pathParameters['routineId']!,
                      ),
                    ),
                    GoRoute(
                      path: 'videos/:videoId',
                      name: 'videoPlayer',
                      builder: (context, state) => VideoPlayerScreen(
                        groupId: state.pathParameters['groupId']!,
                        routineId: state.pathParameters['routineId']!,
                        videoId: state.pathParameters['videoId']!,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ],
    ),
    GoRoute(
      path: '/invite/:token',
      name: 'acceptInvite',
      builder: (context, state) => InviteAcceptScreen(
        token: state.pathParameters['token']!,
      ),
    ),
  ],
);
```

---

## State Management Patterns

Examples using **Riverpod** (adapts easily to Provider too).

```dart
// lib/providers/auth_provider.dart
// Assumes you already have auth token/user stored somewhere
// Adjust to match your existing auth system

final authTokenProvider = Provider<String?>((ref) {
  // Return token from your auth system
  return null; // TODO: Hook to real auth
});

final currentUserIdProvider = Provider<String?>((ref) {
  // Return current user ID from your auth system
  return null; // TODO: Hook to real auth
});

// ============== API CLIENT PROVIDER ==============

final apiClientProvider = Provider<ApiClient>((ref) {
  final authToken = ref.watch(authTokenProvider);
  return ApiClient(
    baseUrl: 'http://localhost:8000',
    authToken: authToken,
  );
});

// ============== GROUPS ==============

// lib/providers/groups_provider.dart

final groupsProvider = FutureProvider<List<Group>>((ref) async {
  final api = ref.watch(apiClientProvider);
  return api.getGroups();
});

final groupDetailProvider = FutureProvider.family<GroupDetail, String>(
  (ref, groupId) async {
    final api = ref.watch(apiClientProvider);
    return api.getGroupWithMembers(groupId);
  },
);

// Manage create group state
class _CreateGroupState {
  final bool isLoading;
  final String? error;
  final Group? createdGroup;

  _CreateGroupState({
    this.isLoading = false,
    this.error,
    this.createdGroup,
  });

  _CreateGroupState copyWith({
    bool? isLoading,
    String? error,
    Group? createdGroup,
  }) =>
    _CreateGroupState(
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
      createdGroup: createdGroup ?? this.createdGroup,
    );
}

class _CreateGroupNotifier extends StateNotifier<_CreateGroupState> {
  final ApiClient _api;
  final Ref _ref;

  _CreateGroupNotifier(this._api, this._ref) : super(_CreateGroupState());

  Future<bool> createGroup({
    required String name,
    required String description,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final group = await _api.createGroup(name: name, description: description);
      state = state.copyWith(isLoading: false, createdGroup: group);
      // Invalidate groups list to refresh
      _ref.refresh(groupsProvider);
      return true;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to create group: $e',
      );
      return false;
    }
  }
}

final createGroupProvider = StateNotifierProvider<_CreateGroupNotifier, _CreateGroupState>(
  (ref) => _CreateGroupNotifier(ref.watch(apiClientProvider), ref),
);

// ============== ROUTINES ==============

// lib/providers/routines_provider.dart

final routinesProvider = FutureProvider.family<List<Routine>, String>(
  (ref, groupId) async {
    final api = ref.watch(apiClientProvider);
    return api.getRoutines(groupId);
  },
);

final routineDetailProvider = FutureProvider.family<Routine, (String, String)>(
  (ref, args) async {
    final api = ref.watch(apiClientProvider);
    return api.getRoutine(args.$1, args.$2);
  },
);

class _CreateRoutineState {
  final bool isLoading;
  final String? error;
  final Routine? createdRoutine;

  _CreateRoutineState({
    this.isLoading = false,
    this.error,
    this.createdRoutine,
  });

  _CreateRoutineState copyWith({
    bool? isLoading,
    String? error,
    Routine? createdRoutine,
  }) =>
    _CreateRoutineState(
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
      createdRoutine: createdRoutine ?? this.createdRoutine,
    );
}

class _CreateRoutineNotifier extends StateNotifier<_CreateRoutineState> {
  final ApiClient _api;
  final String _groupId;
  final Ref _ref;

  _CreateRoutineNotifier(this._api, this._groupId, this._ref)
    : super(_CreateRoutineState());

  Future<bool> createRoutine({
    required String name,
    required String description,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final routine = await _api.createRoutine(
        groupId: _groupId,
        name: name,
        description: description,
      );
      state = state.copyWith(isLoading: false, createdRoutine: routine);
      // Invalidate routines list
      _ref.refresh(routinesProvider(_groupId));
      return true;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to create routine: $e',
      );
      return false;
    }
  }
}

final createRoutineProvider = StateNotifierProvider.family<
  _CreateRoutineNotifier,
  _CreateRoutineState,
  String
>(
  (ref, groupId) => _CreateRoutineNotifier(
    ref.watch(apiClientProvider),
    groupId,
    ref,
  ),
);

// ============== VIDEOS ==============

// lib/providers/videos_provider.dart

final videosProvider = FutureProvider.family<List<RoutineVideo>, (String, String)>(
  (ref, args) async {
    final api = ref.watch(apiClientProvider);
    return api.getVideos(groupId: args.$1, routineId: args.$2);
  },
);

class _VideoUploadState {
  final bool isRegisteringUpload;
  final bool isUploading;
  final double uploadProgress;
  final String? error;
  final String? successVideoId;

  _VideoUploadState({
    this.isRegisteringUpload = false,
    this.isUploading = false,
    this.uploadProgress = 0.0,
    this.error,
    this.successVideoId,
  });

  _VideoUploadState copyWith({
    bool? isRegisteringUpload,
    bool? isUploading,
    double? uploadProgress,
    String? error,
    String? successVideoId,
  }) =>
    _VideoUploadState(
      isRegisteringUpload: isRegisteringUpload ?? this.isRegisteringUpload,
      isUploading: isUploading ?? this.isUploading,
      uploadProgress: uploadProgress ?? this.uploadProgress,
      error: error ?? this.error,
      successVideoId: successVideoId ?? this.successVideoId,
    );
}

class _VideoUploadNotifier extends StateNotifier<_VideoUploadState> {
  final ApiClient _api;
  final Ref _ref;

  _VideoUploadNotifier(this._api, this._ref) : super(_VideoUploadState());

  Future<bool> uploadVideo({
    required String groupId,
    required String routineId,
    required File videoFile,
    required String fileName,
  }) async {
    state = state.copyWith(isRegisteringUpload: true, error: null);

    try {
      // Step 1: Register upload
      final fileSize = (await videoFile.length()).toString();
      final registration = await _api.registerVideoUpload(
        groupId: groupId,
        routineId: routineId,
        fileName: fileName,
        fileSize: fileSize,
      );

      state = state.copyWith(isRegisteringUpload: false, isUploading: true);

      // Step 2: Upload file to presigned URL
      await _api.uploadVideoFile(
        presignedUrl: registration.presignedUrl,
        file: videoFile,
        onProgress: (progress) {
          state = state.copyWith(uploadProgress: progress);
        },
      );

      // Step 3: Finalize upload
      await _api.finalizeVideoUpload(
        groupId: groupId,
        routineId: routineId,
        videoId: registration.video.id,
      );

      state = state.copyWith(
        isUploading: false,
        uploadProgress: 0.0,
        successVideoId: registration.video.id,
      );

      // Refresh videos list
      _ref.refresh(videosProvider((groupId, routineId)));
      return true;
    } catch (e) {
      state = state.copyWith(
        isRegisteringUpload: false,
        isUploading: false,
        uploadProgress: 0.0,
        error: 'Upload failed: $e',
      );
      return false;
    }
  }

  void reset() {
    state = _VideoUploadState();
  }
}

final videoUploadProvider = StateNotifierProvider<
  _VideoUploadNotifier,
  _VideoUploadState
>(
  (ref) => _VideoUploadNotifier(ref.watch(apiClientProvider), ref),
);

// ============== INVITES ==============

// lib/providers/invites_provider.dart

final pendingInvitesProvider = FutureProvider<List<GroupInvite>>((ref) async {
  final api = ref.watch(apiClientProvider);
  return api.getPendingInvites();
});

class _AcceptInviteState {
  final bool isLoading;
  final String? error;
  final Group? acceptedGroup;

  _AcceptInviteState({
    this.isLoading = false,
    this.error,
    this.acceptedGroup,
  });

  _AcceptInviteState copyWith({
    bool? isLoading,
    String? error,
    Group? acceptedGroup,
  }) =>
    _AcceptInviteState(
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
      acceptedGroup: acceptedGroup ?? this.acceptedGroup,
    );
}

class _AcceptInviteNotifier extends StateNotifier<_AcceptInviteState> {
  final ApiClient _api;
  final Ref _ref;

  _AcceptInviteNotifier(this._api, this._ref) : super(_AcceptInviteState());

  Future<bool> acceptInvite(String token) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final group = await _api.acceptGroupInvite(token);
      state = state.copyWith(isLoading: false, acceptedGroup: group);
      // Refresh groups list
      _ref.refresh(groupsProvider);
      return true;
    } catch (e) {
      // Use generic error message for security/privacy
      state = state.copyWith(
        isLoading: false,
        error: 'Invite is invalid or expired.',
      );
      return false;
    }
  }
}

final acceptInviteProvider = StateNotifierProvider<
  _AcceptInviteNotifier,
  _AcceptInviteState
>(
  (ref) => _AcceptInviteNotifier(ref.watch(apiClientProvider), ref),
);

// ============== NOTES ==============

// lib/providers/notes_provider.dart

final routineNotesProvider = FutureProvider.family<
  List<RoutineNote>,
  (String, String)
>(
  (ref, args) async {
    final api = ref.watch(apiClientProvider);
    return api.getRoutineNotes(groupId: args.$1, routineId: args.$2);
  },
);

final videoNotesProvider = FutureProvider.family<
  List<RoutineNote>,
  (String, String, String)
>(
  (ref, args) async {
    final api = ref.watch(apiClientProvider);
    return api.getVideoNotes(
      groupId: args.$1,
      routineId: args.$2,
      videoId: args.$3,
    );
  },
);

class _AddNoteState {
  final bool isLoading;
  final String? error;
  final RoutineNote? addedNote;

  _AddNoteState({
    this.isLoading = false,
    this.error,
    this.addedNote,
  });

  _AddNoteState copyWith({
    bool? isLoading,
    String? error,
    RoutineNote? addedNote,
  }) =>
    _AddNoteState(
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
      addedNote: addedNote ?? this.addedNote,
    );
}

class _AddNoteNotifier extends StateNotifier<_AddNoteState> {
  final ApiClient _api;
  final Ref _ref;

  _AddNoteNotifier(this._api, this._ref) : super(_AddNoteState());

  Future<bool> addRoutineNote({
    required String groupId,
    required String routineId,
    required String content,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final note = await _api.addRoutineNote(
        groupId: groupId,
        routineId: routineId,
        content: content,
      );
      state = state.copyWith(isLoading: false, addedNote: note);
      _ref.refresh(routineNotesProvider((groupId, routineId)));
      return true;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to add note: $e',
      );
      return false;
    }
  }

  Future<bool> addVideoNote({
    required String groupId,
    required String routineId,
    required String videoId,
    required String content,
    int? videoTimestampMs,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final note = await _api.addVideoNote(
        groupId: groupId,
        routineId: routineId,
        videoId: videoId,
        content: content,
        videoTimestampMs: videoTimestampMs,
      );
      state = state.copyWith(isLoading: false, addedNote: note);
      _ref.refresh(videoNotesProvider((groupId, routineId, videoId)));
      return true;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to add note: $e',
      );
      return false;
    }
  }
}

final addNoteProvider = StateNotifierProvider<
  _AddNoteNotifier,
  _AddNoteState
>(
  (ref) => _AddNoteNotifier(ref.watch(apiClientProvider), ref),
);
```

---

## Screen/Page Structure

Key screen examples:

```dart
// lib/screens/groups/group_detail_screen.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../models/group.dart';
import '../../providers/groups_provider.dart';

class GroupDetailScreen extends ConsumerStatefulWidget {
  final String groupId;

  const GroupDetailScreen({
    required this.groupId,
    Key? key,
  }) : super(key: key);

  @override
  ConsumerState<GroupDetailScreen> createState() => _GroupDetailScreenState();
}

class _GroupDetailScreenState extends ConsumerState<GroupDetailScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final groupAsync = ref.watch(groupDetailProvider(widget.groupId));

    return Scaffold(
      appBar: AppBar(
        title: groupAsync.maybeWhen(
          data: (data) => Text(data.group.name),
          orElse: () => const Text('Group'),
        ),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'Routines'),
            Tab(text: 'Members'),
            Tab(text: 'Invites'),
          ],
        ),
      ),
      body: groupAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, stack) => Center(
          child: Text('Error loading group: $error'),
        ),
        data: (data) {
          return TabBarView(
            controller: _tabController,
            children: [
              _RoutinesTab(groupId: widget.groupId, group: data.group),
              _MembersTab(members: data.members),
              _InvitesTab(groupId: widget.groupId),
            ],
          );
        },
      ),
      floatingActionButton: _tabController.index == 0
        ? FloatingActionButton(
            onPressed: () {
              context.push('/groups/${widget.groupId}/routines/create');
            },
            child: const Icon(Icons.add),
          )
        : null,
    );
  }
}

class _RoutinesTab extends ConsumerWidget {
  final String groupId;
  final Group group;

  const _RoutinesTab({
    required this.groupId,
    required this.group,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final routinesAsync = ref.watch(routinesProvider(groupId));

    return routinesAsync.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (error, stack) => Center(
        child: Text('Error loading routines: $error'),
      ),
      data: (routines) {
        if (routines.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.theaters, size: 64, color: Colors.grey),
                const SizedBox(height: 16),
                const Text('No routines yet'),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: () {
                    context.push('/groups/$groupId/routines/create');
                  },
                  child: const Text('Create Routine'),
                ),
              ],
            ),
          );
        }

        return ListView.builder(
          itemCount: routines.length,
          itemBuilder: (context, index) {
            final routine = routines[index];
            return ListTile(
              title: Text(routine.name),
              subtitle: Text(routine.description),
              trailing: Text('${routine.videoCount} videos'),
              onTap: () {
                context.push('/groups/$groupId/routines/${routine.id}');
              },
            );
          },
        );
      },
    );
  }
}

class _MembersTab extends StatelessWidget {
  final List<GroupMember> members;

  const _MembersTab({required this.members});

  @override
  Widget build(BuildContext context) {
    if (members.isEmpty) {
      return const Center(child: Text('No members yet'));
    }

    return ListView.builder(
      itemCount: members.length,
      itemBuilder: (context, index) {
        final member = members[index];
        return ListTile(
          title: Text(member.displayName ?? member.email),
          subtitle: Text(member.email),
          trailing: Chip(label: Text(member.role)),
        );
      },
    );
  }
}

class _InvitesTab extends ConsumerWidget {
  final String groupId;

  const _InvitesTab({required this.groupId});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ElevatedButton(
            onPressed: () {
              context.push('/groups/$groupId/invites');
            },
            child: const Text('Send Invite'),
          ),
        ],
      ),
    );
  }
}
```

---

## End-to-End Flow Examples

### Flow: Upload Video (3-Step Process)

```dart
// lib/screens/routines/video_upload_screen.dart

class VideoUploadScreen extends ConsumerStatefulWidget {
  final String groupId;
  final String routineId;

  const VideoUploadScreen({
    required this.groupId,
    required this.routineId,
    Key? key,
  }) : super(key: key);

  @override
  ConsumerState<VideoUploadScreen> createState() => _VideoUploadScreenState();
}

class _VideoUploadScreenState extends ConsumerState<VideoUploadScreen> {
  File? _selectedFile;

  Future<void> _pickVideo() async {
    // Use image_picker or file_picker package
    // final result = await FilePicker.platform.pickFiles(type: FileType.video);
    // if (result != null) {
    //   setState(() => _selectedFile = File(result.files.single.path!));
    // }
  }

  Future<void> _startUpload() async {
    if (_selectedFile == null) return;

    final notifier = ref.read(videoUploadProvider.notifier);
    final success = await notifier.uploadVideo(
      groupId: widget.groupId,
      routineId: widget.routineId,
      videoFile: _selectedFile!,
      fileName: _selectedFile!.path.split('/').last,
    );

    if (success && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Video uploaded successfully!')),
      );
      context.pop();
    }
  }

  @override
  Widget build(BuildContext context) {
    final uploadState = ref.watch(videoUploadProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Upload Video')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_selectedFile != null)
              Text('Selected: ${_selectedFile!.path.split('/').last}')
            else
              const Text('No video selected'),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: uploadState.isRegisteringUpload || uploadState.isUploading
                ? null
                : _pickVideo,
              child: const Text('Select Video'),
            ),
            const SizedBox(height: 32),
            if (uploadState.error != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade100,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(uploadState.error!),
              ),
            const SizedBox(height: 16),
            if (uploadState.isRegisteringUpload)
              const Text('Preparing upload...'),
            if (uploadState.isUploading) ...[
              Text('Uploading: ${(uploadState.uploadProgress * 100).toStringAsFixed(1)}%'),
              const SizedBox(height: 8),
              LinearProgressIndicator(value: uploadState.uploadProgress),
            ],
            const Spacer(),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed:
                  _selectedFile == null || uploadState.isRegisteringUpload || uploadState.isUploading
                    ? null
                    : _startUpload,
                child: const Text('Upload'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
```

---

## Error Handling & UX

Privacy-first error handling:

```dart
// lib/utils/error_handler.dart

String handleApiError(Object error) {
  if (error is UnauthorizedException) {
    return 'Session expired. Please log in again.';
  } else if (error is NotFoundException) {
    return 'Not found or you don\'t have access.';
  } else if (error is InvalidInviteException) {
    return 'Invite is invalid or expired.';
  } else {
    return 'Something went wrong. Please try again.';
  }
}

// In your screens:
if (someErrorState != null) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Text(handleApiError(someErrorState)),
      backgroundColor: Colors.red,
    ),
  );
}
```

---

## Testing Strategy

```dart
// test/services/api_client_test.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

void main() {
  group('ApiClient', () {
    test('createGroup returns Group on success', () async {
      // Mock HTTP and test
    });

    test('acceptGroupInvite throws InvalidInviteException on 400', () async {
      // Mock HTTP and test
    });

    test('uploadVideoFile reports progress', () async {
      // Mock file and HTTP stream
    });
  });
}
```

---

## Implementation Checklist

- [ ] Create data models (Group, Routine, RoutineVideo, RoutineNote, GroupInvite)
- [ ] Implement ApiClient service with all CRUD operations
- [ ] Set up navigation routes (groups → routines → videos)
- [ ] Create Riverpod/Provider state management providers
- [ ] Build GroupDetailScreen with tabbed interface
- [ ] Build RoutineDetailScreen (videos + notes)
- [ ] Build VideoUploadScreen (3-step: register → upload → finalize)
- [ ] Build VideoPlayerScreen (with notes timeline)
- [ ] Build InviteAcceptScreen (auto-accept on load)
- [ ] Implement privacy-first error handling (non-leaky 404)
- [ ] Add unit tests for ApiClient
- [ ] Test upload progress reporting
- [ ] Verify pending uploads don't leak to other members
- [ ] Test soft-delete behavior (video_deleted flag on notes)

---

## Summary

This guide provides:

1. **Data models** matching the backend API
2. **Complete ApiClient** with all endpoints
3. **Navigation setup** (Go Router example)
4. **State management** patterns for Riverpod
5. **Screen implementations** for key flows
6. **Privacy-first error handling**
7. **Testing patterns**

Adapt package names, patterns, and naming to match your existing codebase. All core logic translates directly to other state management frameworks (Provider, GetX, BLoC).
